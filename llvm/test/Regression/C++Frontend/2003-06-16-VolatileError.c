/* This crashes the CFE.  */
extern int volatile test;
int volatile test = 0;

int main() { return 0; }
