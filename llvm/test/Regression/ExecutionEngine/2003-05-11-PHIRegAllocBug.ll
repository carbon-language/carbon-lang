target endian = little
target pointersize = 32

implementation

int %main() {
entry:
	br label %endif
then:
	br label %endif
endif:
	%x = phi uint [ 4, %entry ], [ 27, %then ]
	%result = phi int [ 32, %then ], [ 0, %entry ]
	ret int 0
}
