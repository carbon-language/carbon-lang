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
LCOV_FILE = ./.build/coverage.lcov
SHELL=/bin/bash

build: ${GRAMMAR}.swift
	swift build ${SWIFT_FLAGS}

test: ${GRAMMAR}.swift
	swift test ${SWIFT_FLAGS}

test-lcov: ${GRAMMAR}.swift
	swift build --build-tests --enable-code-coverage
	$$(swift test --enable-code-coverage --verbose ${SWIFT_FLAGS} 2>&1 \
	   | tee /dev/tty | grep 'llvm-cov export' \
	   | sed -e 's/ export / export -format=lcov /') > "${LCOV_FILE}"

test-jcov: ${GRAMMAR}.swift
	swift test --enable-code-coverage ${SWIFT_FLAGS}

${CITRON}: ${CITRON_SRC}/citron.c
	mkdir -p ${BIN} && ${CC} $^ -o $@

clean:
	rm -rf ${GRAMMAR}.swift ./.build

${GRAMMAR}.swift: ${CITRON} ${GRAMMAR}.citron
	rm -f $@
	${CITRON} ${GRAMMAR}.citron -o $@
	chmod -w $@
