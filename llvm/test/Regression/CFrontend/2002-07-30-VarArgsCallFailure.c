#include <stdio.h>
int tcount;
void foo() {
	char Buf[10];
	sprintf(Buf, "n%%%d", tcount++);
}
