%y = weak global sbyte 0
implementation
uint %testcaseshr() {
entry:
	ret uint shr (uint cast (sbyte* %y to uint), ubyte 4)
}
uint %testcaseshl() {
entry:
	ret uint shl (uint cast (sbyte* %y to uint), ubyte 4)
}
