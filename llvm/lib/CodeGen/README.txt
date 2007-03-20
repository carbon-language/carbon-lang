Common register allocation / spilling problem:

	mul lr, r4, lr
	str lr, [sp, #+52]
	ldr lr, [r1, #+32]
	sxth r3, r3
	ldr r4, [sp, #+52]
	mla r4, r3, lr, r4

can be:

	mul lr, r4, lr
        mov r4, lr
	str lr, [sp, #+52]
	ldr lr, [r1, #+32]
	sxth r3, r3
	mla r4, r3, lr, r4

and then "merge" mul and mov:

	mul r4, r4, lr
	str lr, [sp, #+52]
	ldr lr, [r1, #+32]
	sxth r3, r3
	mla r4, r3, lr, r4

It also increase the likelyhood the store may become dead.
