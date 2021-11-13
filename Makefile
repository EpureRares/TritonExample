build:
	gcc -o example example.c

run:
	python3 emulate.py --binaryPath "example"
