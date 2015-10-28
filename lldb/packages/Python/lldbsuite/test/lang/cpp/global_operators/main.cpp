struct Struct {
	int value;
};

bool operator==(const Struct &a, const Struct &b) {
	return a.value == b.value;
}

int main() {
	Struct s1, s2, s3;
	s1.value = 3;
	s2.value = 5;
	s3.value = 3;
	return 0; // break here
}

