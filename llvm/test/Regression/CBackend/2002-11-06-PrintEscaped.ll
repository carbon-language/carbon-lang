%testString = internal constant [18 x sbyte] c "Escaped newline\n\00"

implementation

declare int %printf(sbyte*, ...)

int %main() {
  call int (sbyte*, ...)* %printf( sbyte* getelementptr ([18 x sbyte]* %testString, long 0, long 0))
  ret int 0
}
