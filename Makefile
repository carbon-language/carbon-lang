GRAMMAR = ./Sources/Syntax/Parser
CC = clang
SWIFTC = swiftc
BIN = ./bin
CITRON_SRC = ./citron/src
CITRON = ${BIN}/citron

build: ${GRAMMAR}.swift
	swift test

test: ${GRAMMAR}.swift
	swift test

${CITRON}: ${CITRON_SRC}/citron.c
	mkdir -p ${BIN} && ${CC} $^ -o $@

clean:
	rm ${GRAMMAR}.swift && swift clean

${GRAMMAR}.swift: ${CITRON} ${GRAMMAR}.citron
	${CITRON} ${GRAMMAR}.citron -o $@
