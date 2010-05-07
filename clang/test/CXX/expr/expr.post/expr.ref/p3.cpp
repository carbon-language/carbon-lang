// RUN: %clang_cc1 -verify -fsyntax-only %s

template<typename T> struct Node {
	int lhs;
	void splay( )                
	{
		Node<T> n[1];
		(void)n->lhs;
	}
};

void f() {
	Node<int> n;
	return n.splay();
}
