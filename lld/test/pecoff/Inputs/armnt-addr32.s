
@ static const int i = 0;
@ const int * const is[] = { &i, };

	.syntax unified
	.thumb
	.text

	.section .rdata,"rd"
	.align 2	# @i
i:
	.long 0		# 0x0

	.global is	# @is
	.align 2
is:
	.long i

