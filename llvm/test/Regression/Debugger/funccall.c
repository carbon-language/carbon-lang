
static int q;

void foo() {
	int t = q;
	q = t + 1;
}
int main() {
	q = 0;
	foo();
	q = q - 1;

	return q;
}
