#define NUM 128

int A[NUM];
int R;

int not_a_reduction(void) {
	int i;
	for (i = 0; i < NUM; i++) {
		R += 1 + A[i];
	}
	return R;
}
