template<typename> struct MergeSpecializations;
template<typename T> struct MergeSpecializations<T[]> {
  typedef int partially_specialized_in_c;
};
template<> struct MergeSpecializations<bool> {
  typedef int explicitly_specialized_in_c;
};

template<typename T> struct MergeTemplateDefinitions {
  static constexpr int f();
  static constexpr int g();
};
template<typename T> constexpr int MergeTemplateDefinitions<T>::g() { return 2; }
