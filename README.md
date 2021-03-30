# Executable Semantics In Swift

Swift implementation of executable semantics of
[Carbon](https://carbon-language/carbon-lang).

## Preparation

1. Have Swift installed and in your path.
2. `git submodule update --init`

## To Build

    make build
    
## To Test

    make test

## To work on the project in Xcode

    make Sources/Parser.swift
    open Package.swift

Note that if you modify Sources/Parser.citron or when you pull new changes from
GitHub, you'll need to run the `make` command above again before proceeding.
