#ifndef ASSOC_LIST_H
#define ASSOC_LIST_H

#include <iostream>
#include <string>
using std::cerr;
using std::endl;

template<class K, class V>
struct AList {
  K key;
  V value;
  AList* next;
  AList(K k, V v, AList* n) : key(k), value(v), next(n) { }
};

template<class K, class V>
V lookup(int lineno, AList<K,V>* alist, K key, void (*print_key)(K)) {
  if (alist == NULL) {
    cerr << lineno << ": could not find `";
    print_key(key);
    cerr << "`" << endl;
    exit(-1);
  } else if (alist->key == key) {
    return alist->value;
  } else {
    return lookup(lineno, alist->next, key, print_key);
  }
}



#endif
