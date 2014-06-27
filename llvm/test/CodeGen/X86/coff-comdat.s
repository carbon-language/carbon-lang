	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 1
	.def	 _f1;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,_f1
	.globl	_f1
	.align	16, 0x90
_f1:                                    # @f1
# BB#0:
	retl

	.def	 _f2;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",same_contents,_f2
	.globl	_f2
	.align	16, 0x90
_f2:                                    # @f2
# BB#0:
	retl

	.def	 _f3;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",largest,_f3
	.globl	_f3
	.align	16, 0x90
_f3:                                    # @f3
# BB#0:
	retl

	.def	 _f4;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,_f4
	.globl	_f4
	.align	16, 0x90
_f4:                                    # @f4
# BB#0:
	retl

	.def	 _f5;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",same_size,_f5
	.globl	_f5
	.align	16, 0x90
_f5:                                    # @f5
# BB#0:
	retl

	.def	 @v7@0;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",associative,@f7@0
	.globl	@v7@0
	.align	16, 0x90
@v7@0:                                  # @"\01@v7@0"
# BB#0:
	retl

	.def	 @f7@0;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,@f7@0
	.globl	@f7@0
	.align	16, 0x90
@f7@0:                                  # @"\01@f7@0"
# BB#0:
	retl

	.def	 @v8@0;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",associative,@f8@0
	.globl	@v8@0
	.align	16, 0x90
@v8@0:                                  # @v8
# BB#0:
	retl

	.def	 @f8@0;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,@f8@0
	.globl	@f8@0
	.align	16, 0x90
@f8@0:                                  # @f8
# BB#0:
	retl

	.section	.bss,"bw",associative,_f1
	.globl	_v1                     # @v1
	.align	4
_v1:
	.long	0                       # 0x0

	.section	.bss,"bw",associative,_f2
	.globl	_v2                     # @v2
	.align	4
_v2:
	.long	0                       # 0x0

	.section	.bss,"bw",associative,_f3
	.globl	_v3                     # @v3
	.align	4
_v3:
	.long	0                       # 0x0

	.section	.bss,"bw",associative,_f4
	.globl	_v4                     # @v4
	.align	4
_v4:
	.long	0                       # 0x0

	.section	.bss,"bw",associative,_f5
	.globl	_v5                     # @v5
	.align	4
_v5:
	.long	0                       # 0x0

	.section	.bss,"bw",associative,_f6
	.globl	_v6                     # @v6
	.align	4
_v6:
	.long	0                       # 0x0

	.section	.bss,"bw",same_size,_f6
	.globl	_f6                     # @f6
	.align	4
_f6:
	.long	0                       # 0x0

	.section	.rdata,"rd"
	.align	4                       # @some_name
L_some_name:
	.zero	8


	.globl	_vftable
_vftable = L_some_name+4
