GRAMMAR = ./Sources/Syntax/Parser
CC = clang
SWIFTC = swiftc
BIN = ./bin
CITRON_SRC = ./citron/src
CITRON = ${BIN}/citron
SWIFT_FLAGS =

build: ${GRAMMAR}.swift
	swift build ${SWIFT_FLAGS} 

test: ${GRAMMAR}.swift
	swift  test --enable-test-discovery ${SWIFT_FLAGS} 

${CITRON}: ${CITRON_SRC}/citron.c
	mkdir -p ${BIN} && ${CC} $^ -o $@

clean:
	rm ${GRAMMAR}.swift && swift clean

${GRAMMAR}.swift: ${CITRON} ${GRAMMAR}.citron
	rm -f $@
	if ${CITRON} ${GRAMMAR}.citron -o $@ 2>&1 \
	  | grep -E -v '[0-9]+ parsing conflicts.' ; then \
	  rm -f $@; exit 1; fi
	chmod -w $@
