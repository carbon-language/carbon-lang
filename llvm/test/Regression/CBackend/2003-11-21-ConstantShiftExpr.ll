%y = weak global sbyte 0
implementation
uint %testcase() {
entry:
	ret uint shr (uint cast (sbyte* %y to uint), ubyte 4)
}
