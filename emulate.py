from typing import List, Dict, Set
from triton import TritonContext, ARCH, Instruction, MemoryAccess, CPUSIZE, MODE, OPCODE
import logging
import string
import logging
import argparse
import sys
import gdb

Triton = TritonContext()

def parseArgs():
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-bp", "--binaryPath", required=True,
                    help="the test binary location")

    args = ap.parse_args()
    return args


class Tracer:
	# Given a context where to emulate the binary already setup in memory with its input, and the PC address to emulate from, plus a few parameters...
	# Returns a tuple (true if the optional target address was reached, num new basic blocks found - if countBBlocks is True)
	# AND the path of basic block addresses found in this run
	@staticmethod
	def emulate(pc: int, countBBlocks: bool):
		targetAddressFound = False
		currentBBlockAddr = pc  # The basic block address that we started to analyze currently
		numNewBasicBlocks = 0  # The number of new basic blocks found by this function (only if countBBlocks was activated)
		newBasicBlocksFound = set()
		blocksFound: Set[int] = set()
		basicBlocksPathFoundThisRun = []

		def restoreContext():
			Triton.setConcreteRegisterValue(Triton.registers.rax, int(gdb.parse_and_eval('$rax')))
			Triton.setConcreteRegisterValue(Triton.registers.rbx, int(gdb.parse_and_eval('$rbx')))
			Triton.setConcreteRegisterValue(Triton.registers.rcx, int(gdb.parse_and_eval('$rcx')))
			Triton.setConcreteRegisterValue(Triton.registers.rdx, int(gdb.parse_and_eval('$rdx')))
			Triton.setConcreteRegisterValue(Triton.registers.rsp, int(gdb.parse_and_eval('$rsp')))
			Triton.setConcreteRegisterValue(Triton.registers.rbp, int(gdb.parse_and_eval('$rbp')))
			Triton.setConcreteRegisterValue(Triton.registers.rdi, int(gdb.parse_and_eval('$rdi')))
			Triton.setConcreteRegisterValue(Triton.registers.rsi, int(gdb.parse_and_eval('$rsi')))
			Triton.setConcreteRegisterValue(Triton.registers.r8, int(gdb.parse_and_eval('$r8')))
			Triton.setConcreteRegisterValue(Triton.registers.r9, int(gdb.parse_and_eval('$r9')))
			Triton.setConcreteRegisterValue(Triton.registers.r10, int(gdb.parse_and_eval('$r10')))
			Triton.setConcreteRegisterValue(Triton.registers.r11, int(gdb.parse_and_eval('$r11')))
			Triton.setConcreteRegisterValue(Triton.registers.r12, int(gdb.parse_and_eval('$r12')))
			Triton.setConcreteRegisterValue(Triton.registers.r13, int(gdb.parse_and_eval('$r13')))
			Triton.setConcreteRegisterValue(Triton.registers.r14, int(gdb.parse_and_eval('$r14')))
			Triton.setConcreteRegisterValue(Triton.registers.r15, int(gdb.parse_and_eval('$r15')))
			Triton.setConcreteRegisterValue(Triton.registers.rip, int(gdb.parse_and_eval('$rip')))
			Triton.setConcreteRegisterValue(Triton.registers.eflags, int(gdb.parse_and_eval('$eflags')))

		def onBasicBlockFound(addr):
			nonlocal numNewBasicBlocks
			nonlocal newBasicBlocksFound
			nonlocal blocksFound
			nonlocal basicBlocksPathFoundThisRun

			basicBlocksPathFoundThisRun.append(addr)
			# Is this a new basic block ?
			if addr not in blocksFound:
				numNewBasicBlocks += 1
				newBasicBlocksFound.add(addr)
				blocksFound.add(addr)

		onBasicBlockFound(currentBBlockAddr)

		logging.info('[+] Starting emulation.')
		while pc:
			print(f"pc = {hex(pc)}")

			# Fetch opcode
			opcode = Triton.getConcreteMemoryAreaValue(pc, 16)
			print("RIP register")
			# print(f"{(gdb.parse_and_eval('$rip'))} = {hex(int(gdb.parse_and_eval('$rip')))}")
			# pc = (int(gdb.parse_and_eval('$rip')))
			# Create the ctx instruction
			instruction = Instruction()
			instruction.setOpcode(opcode)
			instruction.setAddress(pc)
			# Process

			Triton.processing(instruction)
			print("Instruction: "+ str(instruction))
			logging.info(instruction)

			# Next
			prevpc = pc
			pc = Triton.getConcreteRegisterValue(Triton.registers.rip)
			
			if prevpc == pc:
				pc += 4;

			if not ((pc >= 0x7ffff7dbe000 and pc <= 0x7ffff7fac000) or pc <= 0x5000):
				gdb.execute("nexti")
				pc = (int(gdb.parse_and_eval('$rip')))
				restoreContext()
			else:
				gdb.execute("stepi")

				if instruction.isControlFlow():
					if pc >= 0x5000:
						pc = (int(gdb.parse_and_eval('$rip')))
						restoreContext()
					currentBBlockAddr = pc
					onBasicBlockFound(currentBBlockAddr)
					logging.info(f"Instruction is control flow of type {instruction.getType()}. Addr of the new Basic block {hex(currentBBlockAddr)}")
			print("---------------------------")
		logging.info('[+] Emulation done.')

		if basicBlocksPathFoundThisRun[-1] == 0: # ret instruction
			basicBlocksPathFoundThisRun = basicBlocksPathFoundThisRun[:-1]
		return targetAddressFound, numNewBasicBlocks, basicBlocksPathFoundThisRun

	# Load the binary segments into the given set of contexts given as a list
	@staticmethod
	def loadBinary(tracersInstances, binaryPath, entryfuncName):
		outEntryFuncAddr = None
		gdb.execute("start");
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

		phdrs = binary.segments
		for phdr in phdrs:
			size = phdr.physical_size
			vaddr = phdr.virtual_address
			logging.info('[+] Loading 0x%06x - 0x%06x' % (vaddr, vaddr + size))
			Triton.setConcreteMemoryAreaValue(vaddr, phdr.content)
				#assert False, "Check where is stack and heap and reset them "

		Tracer.makeRelocation(binary, Triton)
		return binary, outEntryFuncAddr

	@staticmethod
	def makeRelocation(binary, tritonContext):
		import lief

		libc = lief.parse("/lib/x86_64-linux-gnu/libc.so.6")
		libld = lief.parse("/lib64/ld-linux-x86-64.so.2")
		phdrs  = libc.segments
		for phdr in phdrs:
			size = phdr.physical_size
			vaddr  = BASE_LIBC + phdr.virtual_address
			print('Loading 0x%06x - 0x%06x' %(vaddr, vaddr+size))
			Triton.setConcreteMemoryAreaValue(vaddr, phdr.content)

		phdrs  = libld.segments
		for phdr in phdrs:
			size = phdr.physical_size
			vaddr  = BASE_LIBC + phdr.virtual_address
			print('Loading 0x%06x - 0x%06x' %(vaddr, vaddr+size))
			Triton.setConcreteMemoryAreaValue(vaddr, phdr.content)

		let_bind = [
			"printf",
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
				Triton.setConcreteMemoryValue(MemoryAccess(symbolRelo, CPUSIZE.QWORD), BASE_LIBC + libc_sym_addr)

		return

# Put the last bytes as fake sentinel inputs to promote some usages detection outside buffer
# Some constants
# Where the input buffer will reside in the emulated program
INPUT_BUFFER_ADDRESS = 0x10000000

# Memory mapping
BASE_PLT   = 0x10000000
BASE_ARGV  = 0x20000000
BASE_ALLOC = 0x30000000

BASE_LIBC  = 0x7ffff7dbe000

BASE_STACK = 0x9fffffff

# Allocation information used by malloc()
mallocCurrentAllocation = 0
mallocMaxAllocation     = 2048
mallocBase              = BASE_ALLOC
mallocChunkSize         = 0x00010000
fdes = None

if __name__ == '__main__':

	# Set the architecture
    Triton.setArchitecture(ARCH.X86_64)

    # Set a symbolic optimization mode
    Triton.setMode(MODE.ALIGNED_MEMORY, True)
    
    # args = parseArgs()

    # Load the binary info into the given list of tracers. We do this strage API to load only once the binary...
    binary, entryPoint = Tracer.loadBinary(Triton, "example", "main")

    # Define a fake stack
    Triton.setConcreteRegisterValue(Triton.registers.rbp, BASE_STACK)
    Triton.setConcreteRegisterValue(Triton.registers.rsp, BASE_STACK)
    
    Tracer.emulate(entryPoint, False)
    sys.exit(0)