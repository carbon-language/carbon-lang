target endian = little
target pointersize = 32

implementation

int %main(int, sbyte**) {
entry:
	br bool false, label %then, label %endif
then:
	br label %endif
endif:
	%x.0 = phi uint [ 4, %entry ], [ 27, %then ]
	%result.0 = phi int [ 32, %then ], [ 0, %entry ]
	ret int 0
}
