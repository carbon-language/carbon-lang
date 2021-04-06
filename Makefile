# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This Makefile is a thin layer on top of Swift package manager that handles the
# building of the citron parser generator binary and the generation of
# Sources/Parser.swift from Sources/Parser.citron.
GRAMMAR = ./Sources/Parser
CC = cc
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
