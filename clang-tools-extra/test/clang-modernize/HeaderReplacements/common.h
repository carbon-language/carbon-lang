#ifndef CPP11_MIGRATE_TEST_HEADER_REPLACEMENTS_COMMON_H
#define CPP11_MIGRATE_TEST_HEADER_REPLACEMENTS_COMMON_H

struct container {
  struct iterator {
    int &operator*();
    const int &operator*() const;
    iterator &operator++();
    bool operator!=(const iterator &other);
  };

  iterator begin();
  iterator end();
  void push_back(const int &);
};

void func1(int &I);
void func2();

void dostuff() {
  container C;
  for (container::iterator I = C.begin(), E = C.end(); I != E; ++I) {
    func1(*I);
  }
}

#endif // CPP11_MIGRATE_TEST_HEADER_REPLACEMENTS_COMMON_H
