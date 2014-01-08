@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readobj -u - \
@ RUN:   | FileCheck %s

	.syntax unified

	.cpu cortex-a8
	.fpu neon

	.section .personality

	.type __personality,%function
__personality:
	.fnstart
	bkpt
	.fnend


	.section .personality0

	.type personality0,%function
personality0:
	.fnstart
	bx lr
	.fnend


	.section .personality1

	.type personality1,%function
personality1:
	.fnstart
	.pad #0x100
	sub sp, sp, #0x100
	.save {r0-r11}
	push {r0-r11}
	pop {r0-r11}
	add sp, sp, #0x100
	bx lr
	.fnend


	.section .custom_personality

	.type custom_personality,%function
custom_personality:
	.fnstart
	.personality __personality
	bx lr
	.fnend


	.section .opcodes

	.type opcodes,%function
opcodes:
	.fnstart
	.vsave {d8-d12}
	vpush {d8-d12}
	vpop {d8-d12}
	bx lr
	.fnend


	.section .multiple

	.type function0,%function
function0:
	.fnstart
	bx lr
	.fnend

	.type function1,%function
function1:
	.fnstart
	.personality __personality
	bx lr
	.fnend

	.type function2,%function
function2:
	.fnstart
	bx lr
	.fnend

@ CHECK: UnwindInformation {
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.personality
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ CHECK:         FunctionName: __personality
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         ByteCode [
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.personality0
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ CHECK:         FunctionName: personality0
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         ByteCode [
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.personality1
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ CHECK:         FunctionName: personality1
@ CHECK:         ExceptionHandlingTable: .ARM.extab.personality1
@ CHECK:         TableEntryOffset: 0x0
@ CHECK:         Model: Compact
@ CHECK:         PersonalityIndex: 1
@ CHECK:         ByteCode [
@ CHECK:           Instruction: 0xB1
@ CHECK:           Instruction: 0xF
@ CHECK:           Instruction: 0xA7
@ CHECK:           Instruction: 0x3F
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.custom_personality
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ CHECK:         FunctionName: custom_personality
@ CHECK:         ExceptionHandlingTable: .ARM.extab.custom_personality
@ CHECK:         TableEntryOffset: 0x0
@ CHECK:         Model: Generic
@ CHECK:         PersonalityRoutineAddress: 0x0
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.opcodes
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ CHECK:         FunctionName: opcodes
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         ByteCode [
@ CHECK:           Instruction: 0xC9
@ CHECK:           Instruction: 0x84
@ CHECK:           Instruction: 0xB0
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.multiple
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ CHECK:         FunctionName: function0
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         ByteCode [
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:         ]
@ CHECK:       }
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x4
@ CHECK:         FunctionName: function1
@ CHECK:         ExceptionHandlingTable: .ARM.extab.multiple
@ CHECK:         Model: Generic
@ CHECK:         PersonalityRoutineAddress: 0x0
@ CHECK:       }
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x8
@ CHECK:         FunctionName: function2
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         ByteCode [
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:           Instruction: 0xB0
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK: }

