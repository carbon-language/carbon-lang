@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readobj -t \
@ RUN:   | FileCheck %s

	.syntax unified

	.arm

	.type arm_func,%function
arm_func:
	nop

	.thumb_set alias_arm_func, arm_func

	.thumb

	.type thumb_func,%function
	.thumb_func
thumb_func:
	nop

	.thumb_set alias_thumb_func, thumb_func

	.thumb_set seedless, 0x5eed1e55
	.thumb_set eggsalad, seedless + 0x87788358
	.thumb_set faceless, ~eggsalad + 0xe133c002

	.thumb_set alias_undefined_data, badblood

	.data

	.type badblood,%object
badblood:
	.long 0xbadb100d

	.type bedazzle,%object
bedazzle:
	.long 0xbeda221e

	.text
	.thumb

	.thumb_set alias_defined_data, bedazzle

	.type alpha,%function
alpha:
	nop

        .type beta,%function
beta:
	bkpt

	.thumb_set beta, alpha

	.thumb_set alias_undefined, undefined

@ CHECK: Symbol {
@ CHECK:   Name: alias_arm_func
@ CHECK:   Value: 0x1
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: alias_defined_data
@ CHECK:   Value: 0x5
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: alias_thumb_func
@ CHECK:   Value: 0x5
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: alias_undefined_data
@ CHECK:   Value: 0x0
@ CHECK:   Type: Object
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: alpha
@ CHECK:   Value: 0x6
@ XFAIL-CHECK:   Value: 0x7
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: arm_func
@ CHECK:   Value: 0x0
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: bedazzle
@ CHECK:   Value: 0x4
@ CHECK:   Type: Object
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: beta
@ CHECK:   Value: 0x7
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: eggsalad
@ CHECK:   Value: 0xE665A1AD
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: faceless
@ CHECK:   Value: 0xFACE1E55
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: seedless
@ CHECK:   Value: 0x5EED1E55
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: thumb_func
@ CHECK:   Value: 0x5
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: badblood
@ CHECK:   Value: 0x0
@ CHECK:   Type: Object
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: undefined
@ CHECK:   Value: 0x0
@ CHECK:   Type: None
@ CHECK: }

