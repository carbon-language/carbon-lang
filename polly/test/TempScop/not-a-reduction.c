#define TYPE float
#define NUM 4

TYPE A[NUM];
TYPE B[NUM];
TYPE C[NUM];

void vector_multiply(void) {
	int i;
	for (i = 0; i < NUM; i++) {
		A[i] = B[i] * C[i];
	}
}
