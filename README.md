# Executable Semantics In Swift

Swift implementation of executable semantics of
[Carbon](https://carbon-language/carbon-lang).

## Preparation

1. Have Swift installed and in your path.
2. `git submodule update --init`

## To Build on Mac or Linux

    make build
    
## To Test on Mac or Linux

    make test

## To translate parse errors so that your IDE will recognize them

    make test 2>&1 | sed -Ee 's/.*"(.*)", (.*)\.\.<(.*)\)\).*/\1:\2:{\2-\3}/g'

## To work on the project in Xcode

    make Sources/Parser.swift
    open Package.swift

Note that if you modify Sources/Parser.citron, or when you pull new changes from
GitHub, you'll need to run the `make` command above again before proceeding.

## On Windows

If you install [MinGW](https://sourceforge.net/projects/mingw/), you can use
`make Sources/Parser.swift` followed by `swift build` or `swift test`, with the
same caveat as for Xcode users: if you modify Sources/Parser.citron, and when
you pull new changes from GitHub, you'll need to run the `make` command again
before proceeding.  If you don't want to install MinGW, you should be able to
read the simple Makefile to figure out how to build citron and use it to
"manually" generate Sources/Parser.swift from Sources/Parser.citron.
