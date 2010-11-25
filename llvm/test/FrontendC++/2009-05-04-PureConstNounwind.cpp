// RUN: %llvmgxx -S %s -o - | grep nounwind | count 4
int c(void) __attribute__((const));
int p(void) __attribute__((pure));
int t(void);

int f(void) {
	return c() + p() + t();
}
