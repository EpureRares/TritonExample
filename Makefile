build:
	gcc -o example example.c

run:
	python3 emulate.py --binaryPath "example" -entryfuncName "main" --architecture x64 --maxLen 5 --logLevel CRITICAL --secondsBetweenStats 10
