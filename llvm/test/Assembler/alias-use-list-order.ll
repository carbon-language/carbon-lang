; RUN: verify-uselistorder < %s

; Globals.
@global = global i32 0
@alias.ref1 = global i32* getelementptr inbounds (i32, i32* @alias, i64 1)
@alias.ref2 = global i32* getelementptr inbounds (i32, i32* @alias, i64 1)

; Aliases.
@alias = alias i32* @global
@alias.ref3 = alias i32* getelementptr inbounds (i32, i32* @alias, i64 1)
@alias.ref4 = alias i32* getelementptr inbounds (i32, i32* @alias, i64 1)
