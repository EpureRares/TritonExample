from typing import List, Dict, Set
from triton import TritonContext, ARCH, Instruction, MemoryAccess, CPUSIZE, MODE, OPCODE
import logging
import pdb
import os
import array
import string
from bitstring import BitArray
import heapq
import numpy as np
import logging
import argparse
import time
import sys

SENTINEL_SIZE = 4

def parseArgs():
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-bp", "--binaryPath", required=True,
                    help="the test binary location")
    ap.add_argument("-entryfuncName", "--entryfuncName", required=False, default="RIVERTestOneInput",
                    help="the name of the entry function you want to start the test from. By default the function name is 'RIVERTestOneInput'!", type=str)
    ap.add_argument("-arch", "--architecture", required=True,
                    help="architecture of the executable: ARM32, ARM64, X86, X64 are supported")
    ap.add_argument("-max", "--maxLen", required=True,
                    help="maximum size of input length", type=int)
    ap.add_argument("-logLevel", "--logLevel", required=False, default='CRITICAL',
                    help="set the log level threshold, see the Python logging module documentation for the list of levels. Set it to DEBUG to see everything!", type=str)
    ap.add_argument("-secondsBetweenStats", "--secondsBetweenStats", required=False, default='10',
                    help="the interval (in seconds) between showing new stats", type=int)

    args = ap.parse_args()

    loggingLevel = logging._nameToLevel[args.logLevel]
    logging.basicConfig(level=loggingLevel)  # filename='example.log', # Set DEBUG or INFO if you want to see more

    SECONDS_BETWEEN_STATS = args.secondsBetweenStats

    #assert len(args.defaultObsParams) != 4 # There are 4 default obs types
    args.obs_map = 0 #int(args.defaultObsParams[0])
    args.obs_path = 0 #int(args.defaultObsParams[1])
    args.obs_path_stats = 1 # int(args.defaultObsParams[2])
    args.obs_embedding = 0 #int(args.defaultObsParams[3])

    # Set the architecture
    if args.architecture == "ARM32":
        args.architecture = ARCH.ARM32
    elif args.architecture == "ARM64":
        args.achitecture = ARCH.X86_64
    elif args.architecture == "x86":
        args.architecture = ARCH.X86
    elif args.architecture == "x64":
        args.architecture = ARCH.X86_64
    else:
        assert False, "This architecture is not implemented"
        raise NotImplementedError

    Input.MAX_LEN = args.maxLen

    return args

#  Data structures  to hold inputs
# Currently we keep the input as a dictionary mapping from byte indices to values.
# The motivation for this now is that many times the input are large but only small parts from them are changing...
# usePlainBuffer = true if the input is not sparse, to represent the input indices as an array rather than a full vector
class Input:
    def __init__(self, buffer : Dict[int, any] = None, bound = None , priority = None, usePlainBuffer=False):
        self.buffer = buffer
        self.bound = bound
        self.priority = priority
        self.usePlainBuffer = False

    def __lt__(self, other):
        return self.priority > other.priority

    def __str__(self):
        maxKeysToShow = 10
        keysToShow = sorted(self.buffer)[:maxKeysToShow]
        valuesStrToShow = ' '.join(str(self.buffer[k]) for k in keysToShow)
        strRes = (f"({valuesStrToShow}..bound: {self.bound}, priority: {self.priority})")
        return strRes

    # Apply the changes to the buffer, as given in the dictionary mapping from byte index to the new value
    def applyChanges(self, changes : Dict[int, any]):
        if not self.usePlainBuffer:
            self.buffer.update(changes)
        else:
            for byteIndex,value in changes.items():
                self.buffer[byteIndex] = value


    tokensDictionary = []

    NO_ACTION_INDEX = -1
    MAX_LEN = None # Will be set by user parameters

    def sanityCheck(self):
        # Check 1: is input size in the desired range ?
        assert len(self.buffer) <= Input.MAX_LEN, f"Input obtained is bigger than the maximum length !! Max size set in params was {Input.MAX_LEN} while buffer has currently size {len(self.buffer)}"


# A priority queue data structure for holding inputs by their priority
class InputsWorklist:
    def __init__(self):
        self.internalHeap = []

    def extractInput(self):
        if self.internalHeap:
            next_item = heapq.heappop(self.internalHeap)
            return next_item
        else:
            return None

    def addInput(self, inp: Input):
        heapq.heappush(self.internalHeap, inp)

    def __str__(self):
        str = f"[{' ; '.join(inpStr.__str__() for inpStr in self.internalHeap)}]"
        return str

    def __len__(self):
        return len(self.internalHeap)


# Process the list of inputs to convert to bytes if the input was in a string format
def processSeedDict(seedsDict : List[any]):
    for idx, oldVal in enumerate(seedsDict):
        if isinstance(oldVal, str):
            seedsDict[idx] = str.encode(oldVal)

class RiverTracer:
	# Creates the tracer either with symbolic execution enabled or not
	# And with the given architecture
	# if a targetToReach is used, then the emulation stops when the tracer gets to that address
	def __init__(self, architecture, symbolized, maxInputSize, targetAddressToReach = None):
		self.context = TritonContext(architecture)
		self.symbolized = symbolized
		self.resetSymbolicMemoryAtEachRun = False # KEEP IT FALSE OR TELL CPADURARU WHY YOU DO OTHERWISE
		self.maxInputSize = maxInputSize

		if symbolized is False:
			self.context.enableSymbolicEngine(False)
		assert self.context.isSymbolicEngineEnabled() == symbolized

		# Define some symbolic optimizations - play around with these since maybe there are variations between the used program under test
		self.context.setMode(MODE.ALIGNED_MEMORY, True)
		if symbolized:
			self.context.setMode(MODE.ONLY_ON_SYMBOLIZED, True)

		# The set of basic blocks found so far by this tracer.
		self.allBlocksFound: Set[int] = set()
		self.TARGET_TO_REACH = targetAddressToReach
		self.entryFuncAddr = None # Entry function address
		self.codeSection_begin = None # Where the code section begins and ends
		self.codeSection_end = None

		# Create the cache of symbolic variables if they are to be keep fixed.
		inputMaxLenPlusSentinelSize = self.maxInputSize + SENTINEL_SIZE
		self.symbolicVariablesCache = [None] * inputMaxLenPlusSentinelSize
		if self.resetSymbolicMemoryAtEachRun == False:
			for byteIndex in range(inputMaxLenPlusSentinelSize):
				byteAddr = INPUT_BUFFER_ADDRESS + byteIndex
				symbolicVar = self.context.symbolizeMemory(MemoryAccess(byteAddr, CPUSIZE.BYTE))
				self.symbolicVariablesCache[byteIndex] = symbolicVar

		assert self.resetSymbolicMemoryAtEachRun == True or len(self.symbolicVariablesCache) == inputMaxLenPlusSentinelSize


	def resetPersistentState(self):
		self.allBlocksFound = set()

	# Gets the context of this tracer
	def getContext(self):
		return self.context

	def getAstContext(self):
		return self.context.getAstContext()

	# Given a context where to emulate the binary already setup in memory with its input, and the PC address to emulate from, plus a few parameters...
	# Returns a tuple (true if the optional target address was reached, num new basic blocks found - if countBBlocks is True)
	# AND the path of basic block addresses found in this run
	def __emulate(self, pc: int, countBBlocks: bool):
		targetAddressFound = False
		currentBBlockAddr = pc  # The basic block address that we started to analyze currently
		numNewBasicBlocks = 0  # The number of new basic blocks found by this function (only if countBBlocks was activated)
		newBasicBlocksFound = set()
		basicBlocksPathFoundThisRun = []

		def onBasicBlockFound(addr):
			nonlocal numNewBasicBlocks
			nonlocal newBasicBlocksFound
			nonlocal basicBlocksPathFoundThisRun

			basicBlocksPathFoundThisRun.append(addr)
			# Is this a new basic block ?
			if addr not in self.allBlocksFound:
				numNewBasicBlocks += 1
				newBasicBlocksFound.add(addr)
				self.allBlocksFound.add(addr)

		onBasicBlockFound(currentBBlockAddr)

		logging.info('[+] Starting emulation.')
		while pc:
			print(f" pc = {hex(pc)}")

			# Fetch opcode
			opcode = self.context.getConcreteMemoryAreaValue(pc, 16)

			# Create the ctx instruction
			instruction = Instruction()
			instruction.setOpcode(opcode)
			instruction.setAddress(pc)

			# Process
			self.context.processing(instruction)
			print(instruction)
			logging.info(instruction)

			# Next
			prevpc = pc
			pc = self.context.getConcreteRegisterValue(self.context.registers.rip)

			if instruction.isControlFlow():
				currentBBlockAddr = pc
				onBasicBlockFound(currentBBlockAddr)
				logging.info(f"Instruction is control flow of type {instruction.getType()}. Addr of the new Basic block {hex(currentBBlockAddr)}")

			if self.TARGET_TO_REACH is not None and pc == self.TARGET_TO_REACH:
				targetAddressFound = True

		logging.info('[+] Emulation done.')

		if basicBlocksPathFoundThisRun[-1] == 0: # ret instruction
			basicBlocksPathFoundThisRun = basicBlocksPathFoundThisRun[:-1]
		return targetAddressFound, numNewBasicBlocks, basicBlocksPathFoundThisRun

	def debugShowAllSymbolicVariables(self):
		allSymbolicVariables = self.context.getSymbolicVariables()
		print(f"All symbolic variables: {allSymbolicVariables}")

		for k, v in sorted(self.context.getSymbolicVariables().items()):
			print(k, v)
			varValue = self.context.getConcreteVariableValue(v)
			print(f"Var id {k} name and size: {v} = {varValue}")

	# This function initializes the context memory for further emulation
	def __initContext(self, inputToTry: Input, symbolized: bool):
		assert (self.context.isSymbolicEngineEnabled() == symbolized or symbolized == False), "Making sure that context has exactly the matching requirements for the call, nothing more, nothing less"

		inputToTry.sanityCheck()

		# Clean symbolic state
		if symbolized and self.resetSymbolicMemoryAtEachRun:
			self.context.concretizeAllRegister()
			self.context.concretizeAllMemory()

		# Byte level
		def symbolizeAndConcretizeByteIndex(byteIndex, value, symbolized):
			byteAddr = INPUT_BUFFER_ADDRESS + byteIndex

			if symbolized:
				# If not needed to reset symbolic state, just take the variable from the cache store and set its current value
				if self.resetSymbolicMemoryAtEachRun: # Not used anymore
					self.context.setConcreteMemoryValue(byteAddr, value)
					self.context.symbolizeMemory(MemoryAccess(byteAddr, CPUSIZE.BYTE))
				else:
					self.context.setConcreteVariableValue(self.symbolicVariablesCache[byteIndex], value)
					assert self.context.getConcreteMemoryValue(MemoryAccess(byteAddr, CPUSIZE.BYTE)) == value

		# Continuous area level
		def symbolizeAndConcretizeArea(addr, values):
			if symbolized:
				if self.resetSymbolicMemoryAtEachRun: # Not used anymore
					self.context.setConcreteMemoryAreaValue(addr, values)
					for byteIndex, value in enumerate(values):
						byteAddr = INPUT_BUFFER_ADDRESS + byteIndex
						self.context.symbolizeMemory(MemoryAccess(byteAddr, CPUSIZE.BYTE))
				else:
					# If not needed to reset symbolic state, just take the variable from the cache store and set its current value
					# This will update both the symbolic state and concrete memory
					for byteIndex, value in enumerate(values):
						byteAddr = INPUT_BUFFER_ADDRESS + byteIndex
						self.context.setConcreteVariableValue(self.symbolicVariablesCache[byteIndex], value)


		# Symbolize the input bytes in the input seed.
		# Put all the inputs in the buffer in the emulated program memory
		if inputToTry.usePlainBuffer == True:
			assert isinstance(inputToTry.buffer, list), "The input expected to be a series of bytes in a list "
			inputLen = len(inputToTry.buffer)
			symbolizeAndConcretizeArea(INPUT_BUFFER_ADDRESS, inputToTry.buffer)

		else:
			inputLen = max(inputToTry.buffer.keys()) + 1
			for byteIndex, value in inputToTry.buffer.items():
				symbolizeAndConcretizeByteIndex(byteIndex, value, symbolized)

		if symbolized:
			for sentinelByteIndex in range(inputLen, inputLen + SENTINEL_SIZE):
				symbolizeAndConcretizeByteIndex(sentinelByteIndex, 0, symbolized)

		# Point RDI on our buffer. The address of our buffer is arbitrary. We just need
		# to point the RDI register on it as first argument of our targeted function.
		self.context.setConcreteRegisterValue(self.context.registers.rdi, INPUT_BUFFER_ADDRESS)
		self.context.setConcreteRegisterValue(self.context.registers.rsi, inputLen)

		# Setup fake stack on an abitrary address.
		self.context.setConcreteRegisterValue(self.context.registers.rsp, 0x7fffffff)
		self.context.setConcreteRegisterValue(self.context.registers.rbp, 0x7fffffff)
		return

	def runInput(self, inputToTry : Input, symbolized : bool, countBBlocks : bool):
		# Init context memory
		self.__initContext(inputToTry, symbolized=symbolized)

		# Emulate the binary with the setup memory
		return self.__emulate(self.entryFuncAddr, countBBlocks=countBBlocks)

	def getLastRunPathConstraints(self):
		return self.context.getPathConstraints()

	def resetLastRunPathConstraints(self):
		self.context.clearPathConstraints()

	# Ask for a model to change the input conditions such that a base bath + a branch change condition constraints are met
	# Then put all the changed bytes (map from index to value) in a dictionary
	def solveInputChangesForPath(self, constraint):
		assert self.symbolized == True, "you try to solve inputs using a non-symbolic tracer context !"

		model = self.context.getModel(constraint)
		changes = dict()  # A dictionary  from byte index (relative to input buffer beginning) to the value it has in he model
		for k, v in list(model.items()):
			# Get the symbolic variable assigned to the model
			symVar = self.context.getSymbolicVariable(k)
			# Save the new input as seed.
			byteAddrAccessed = symVar.getOrigin()
			byteAddrAccessed_relativeToInputBuffer = byteAddrAccessed - INPUT_BUFFER_ADDRESS
			changes.update({byteAddrAccessed_relativeToInputBuffer: v.getValue()})

		return changes

	# Load the binary segments into the given set of contexts given as a list
	@staticmethod
	def loadBinary(tracersInstances, binaryPath, entryfuncName):
		outEntryFuncAddr = None

		logging.info(f"Loading the binary at path {binaryPath}..")
		import lief
		binary = lief.parse(binaryPath)
		if binary is None:
			assert False, f"Path to binary not found {binaryPath}"
			exit(0)

		text = binary.get_section(".text")
		codeSection_begin = text.file_offset
		codeSection_end = codeSection_begin + text.size

		if outEntryFuncAddr is None:
			logging.info(f"Findind the exported function of interest {binaryPath}..")
			res = binary.exported_functions
			for function in res:
				if entryfuncName in function.name:
					outEntryFuncAddr = function.address
					logging.info(f"Function of interest found at address {outEntryFuncAddr}")
					break
		assert outEntryFuncAddr != None, "Exported function wasn't found"

		for tracerIndex, tracer in enumerate(tracersInstances):
			tracersInstances[tracerIndex].entryFuncAddr = outEntryFuncAddr
			tracersInstances[tracerIndex].codeSection_begin = codeSection_begin
			tracersInstances[tracerIndex].codeSection_end = codeSection_end

			phdrs = binary.segments
			for phdr in phdrs:
				size = phdr.physical_size
				vaddr = phdr.virtual_address
				logging.info('[+] Loading 0x%06x - 0x%06x' % (vaddr, vaddr + size))
				tracersInstances[tracerIndex].context.setConcreteMemoryAreaValue(vaddr, phdr.content)
				#assert False, "Check where is stack and heap and reset them "

			RiverTracer.makeRelocation(binary, tracersInstances[tracerIndex].context)

	@staticmethod
	def makeRelocation(binary, tritonContext):
		import lief

		# ldd <binary>
		# linux-vdso.so.1 (0x00007ffe3a524000)
		# libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fc961d59000)
		# libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fc9619bb000)
		# libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fc9617a3000)
		# libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fc9613b2000)
		# /lib64/ld-linux-x86-64.so.2 (0x00007fc96212d000)

		libc = lief.parse("/lib/x86_64-linux-gnu/libc.so.6")
		phdrs  = libc.segments
		for phdr in phdrs:
			size = phdr.physical_size
			vaddr  = BASE_LIBC + phdr.virtual_address
			print('Loading 0x%06x - 0x%06x' %(vaddr, vaddr+size))
			tritonContext.setConcreteMemoryAreaValue(vaddr, phdr.content)

		let_bind = [
			"printf",
			"dprintf",
			"strlen",
			"vprintf",
			"psiginfo",
			"strchr"
		]

		relocations = [x for x in binary.pltgot_relocations]
		relocations.extend([x for x in binary.dynamic_relocations])
		# Perform our own relocations
		for rel in relocations:
			symbolName = rel.symbol.name
			symbolRelo = rel.address
			if symbolName in let_bind:
				print(f"Hooking {symbolName}")
				libc_sym_addr = libc.get_symbol(symbolName).value
				print(f"name {symbolName} addr {hex(libc_sym_addr)} res {hex(BASE_LIBC + libc_sym_addr)}")
				tritonContext.setConcreteMemoryValue(MemoryAccess(symbolRelo, CPUSIZE.QWORD), BASE_LIBC + libc_sym_addr)

		return

# The online reconstructed graph that shows possible connections between basic blocks.
# We don't keep inputs or symbolic conditions because we don;t need them so far..just the graph
BlocksGraph : Dict[int, Set[int]] = {} # From basic block to the list of basic blocks and possible links

RECONSTRUCT_BB_GRAPH = False # Enable this if you need the block graph above

def onEdgeDetected(fromAddr, toAddr):
    if fromAddr not in BlocksGraph:
        BlocksGraph[fromAddr] = set()
    BlocksGraph[fromAddr].add(toAddr)

# This function returns a set of new inputs based on the last trace.
def Expand(symbolicTracer : RiverTracer, inputToTry):
    logging.critical(f"Seed injected:, {inputToTry}")

    symbolicTracer.runInput(inputToTry, symbolized=True, countBBlocks=False)

    # Set of new inputs
    inputs : List[Input] = []

    # Get path constraints from the last execution
    PathConstraints = symbolicTracer.getLastRunPathConstraints()

    # Get the astContext
    astCtxt = symbolicTracer.getAstContext()

    # This represents the current path constraint, dummy initialization
    currentPathConstraint = astCtxt.equal(astCtxt.bvtrue(), astCtxt.bvtrue())

    # Go through the path constraints from bound of the input (to prevent backtracking as described in the paper)
    PCLen = len(PathConstraints)
    for pcIndex in range(inputToTry.bound, PCLen):
        pc = PathConstraints[pcIndex]

        # Get all branches
        branches = pc.getBranchConstraints()

        if RECONSTRUCT_BB_GRAPH:
            # Put all detected edges in the graph
            for branch in branches:
                onEdgeDetected(branch['srcAddr'], branch['dstAddr'])

        # If there is a condition on this path (not a direct jump), try to reverse it
        if pc.isMultipleBranches():
            takenAddress = pc.getTakenAddress()
            for branch in branches:
                # Get the constraint of the branch which has been not taken
                if branch['dstAddr'] != takenAddress:
                    #print(branch['constraint'])
                    #expr = astCtxt.unroll(branch['constraint'])
                    #expr = ctx.simplify(expr)

                    # Check if we can change current executed path with the branch changed
                    desiredConstrain = astCtxt.land([currentPathConstraint, branch['constraint']])
                    changes = symbolicTracer.solveInputChangesForPath(desiredConstrain)

                    # Then, if a possible change was detected => create a new input entry and add it to the output list
                    if changes:
                        newInput = copy.deepcopy(inputToTry)
                        newInput.applyChanges(changes)
                        newInput.bound = pcIndex + 1
                        inputs.append(newInput)

        # Update the previous constraints with taken(true) branch to keep the same path initially taken
        currentPathConstraint = astCtxt.land([currentPathConstraint, pc.getTakenPredicate()])

    # Clear the path constraints to be clean at the next execution.
    symbolicTracer.resetLastRunPathConstraints()

    return inputs

# This function starts with a given seed dictionary and does concolic execution starting from it.
def SearchInputs(symbolicTracer, simpleTracer, initialSeedDict, binaryPath, outputEndpoint):
    # Init the worklist with the initial seed dict
    worklist  = InputsWorklist()
    forceFinish = False

    # Put all the the inputs in the seed dictionary to the worklist
    for initialSeed in initialSeedDict:
        inp = Input()
        inp.buffer = {k: v for k, v in enumerate(initialSeed)}
        inp.bound = 0
        inp.priority = 0
        worklist.addInput(inp)

    startTime = currTime = time.time()
    while worklist:
        # Take the first seed
        inputSeed : Input = worklist.extractInput()
        newInputs = Expand(symbolicTracer, inputSeed)

        for newInp in newInputs:
            # Execute the input to detect real issues with it
            issue = ExecuteInputToDetectIssues(binaryPath, newInp)
            if issue != None:
                print(f"{binaryPath} has issues: {issue} on input {newInp}")
                pass

            # Assign this input a priority, and check if the hacked target address was found or not
            targetFound, newInp.priority = ScoreInput(newInp, simpleTracer)

            if targetFound:
                logging.critical(f"The solution to get to the target address is input {newInp}")

                forceFinish = True # No reason to continue...
                break

            # Then put it in the worklist
            worklist.addInput(newInp)

        if forceFinish:
            break

    # currTime = outputStats.UpdateOutputStats(startTime, currTime, collectorTracers=[simpleTracer], forceOutput=True)

def ExecuteInputToDetectIssues(binary_path, input : Input):
    from bugs_detection.test_inputs import sig_str, test_input
    return sig_str.get(test_input(binary_path, bytes([v for v in input.buffer.values()])))

def ScoreInput(newInp : Input, simpleTracer : RiverTracer):
    logging.critical(f"--Scoring input {newInp}")
    targetFound, numNewBlocks, allBBsInThisRun = simpleTracer.runInput(newInp, symbolized=False, countBBlocks=True)
    return targetFound, numNewBlocks # as default, return the bound...


# Put the last bytes as fake sentinel inputs to promote some usages detection outside buffer
# Some constants
# Where the input buffer will reside in the emulated program
INPUT_BUFFER_ADDRESS = 0x10000000

# Memory mapping
BASE_PLT   = 0x10000000
BASE_ARGV  = 0x20000000
BASE_ALLOC = 0x30000000

BASE_LIBC  = 0x7ffff7dbf000
# BASE_LIBC  = 0x7ffff701f000

BASE_STACK = 0x9fffffff

# Allocation information used by malloc()
mallocCurrentAllocation = 0
mallocMaxAllocation     = 2048
mallocBase              = BASE_ALLOC
mallocChunkSize         = 0x00010000
fdes = None

if __name__ == '__main__':

    args = parseArgs()

    # Create two tracers : one symbolic used for detecting path constraints etc, and another one less heavy used only for tracing and scoring purpose
    symbolicTracer  = RiverTracer(symbolized=True,  architecture=args.architecture, maxInputSize=args.maxLen, targetAddressToReach=None)
    simpleTracer    = RiverTracer(symbolized=False, architecture=args.architecture, maxInputSize=args.maxLen, targetAddressToReach=None)

    # Load the binary info into the given list of tracers. We do this strage API to load only once the binary...
    RiverTracer.loadBinary([symbolicTracer, simpleTracer], args.binaryPath, args.entryfuncName)

    # TODO Bogdan: Implement the corpus strategies as defined in https://llvm.org/docs/LibFuzzer.html#corpus, or Random if not given
    initialSeedDict = ["good"] # ["a<9d"]
    processSeedDict(initialSeedDict) # Transform the initial seed dict to bytes instead of chars if needed

    SearchInputs(symbolicTracer=symbolicTracer, simpleTracer=simpleTracer, initialSeedDict=initialSeedDict,
                binaryPath=args.binaryPath, outputEndpoint=None)

    sys.exit(0)

