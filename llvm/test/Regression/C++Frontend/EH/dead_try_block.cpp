// This testcase doesn't actually DO any EH

static void foo() {}
int main() {
	try {
		foo();
		return 0;
	} catch(...) {
		return 1;
	}
}
