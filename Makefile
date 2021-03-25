GRAMMAR = ./Sources/Syntax/Parser
CC = clang
SWIFTC = swiftc
BIN = ./bin
CITRON_SRC = ./citron/src
CITRON = ${BIN}/citron
SWIFT_FLAGS =

build: ${GRAMMAR}.swift
	swift build --enable-test-discovery ${SWIFT_FLAGS} 

test: ${GRAMMAR}.swift
	swift test --enable-test-discovery ${SWIFT_FLAGS} 

${CITRON}: ${CITRON_SRC}/citron.c
	mkdir -p ${BIN} && ${CC} $^ -o $@

clean:
	rm -rf ${GRAMMAR}.swift ./.build

${GRAMMAR}.swift: ${CITRON} ${GRAMMAR}.citron
	rm -f $@
	${CITRON} -i ${GRAMMAR}.citron -o $@
	chmod -w $@
