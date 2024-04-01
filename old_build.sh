#!/bin/sh

set -xe

gcc -Wall -Wextra -o twice twice.c -lm
gcc -Wall -Wextra -o gates gates.c -lm
gcc -Wall -Wextra -o nand nand.c -lm