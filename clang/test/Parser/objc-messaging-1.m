// RUN: clang %s -parse-noop
int main ()
{
	int i,j;
	struct S *p;
	[p ii];
	[p if: 1 :2];
	[p inout: 1 :2 another:(2,3,4)];
	[p inout: 1 :2 another:(2,3,4), 6,6,8];
	[p inout: 1 :2 another:(2,3,4), (6,4,5),6,8];
	[p inout: 1 :2 another:(i+10), (i,j-1,5),6,8];
	[p long: 1 :2 another:(i+10), (i,j-1,5),6,8];
	[p : "Hello\n" :2 another:(i+10), (i,j-1,5),6,8];
}
