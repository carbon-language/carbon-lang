#define NUM 128

int A[NUM];
int R;

int reduction(void) {
	int i;
	for (i = 0; i < NUM; i++) {
		R -= A[i];
	}
	return R;
}
