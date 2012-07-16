//===------ AMDILAlgorithms.tpp - AMDIL Template Algorithms Header --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides templates algorithms that extend the STL algorithms, but
// are useful for the AMDIL backend
//
//===----------------------------------------------------------------------===//

// A template function that loops through the iterators and passes the second
// argument along with each iterator to the function. If the function returns
// true, then the current iterator is invalidated and it moves back, before
// moving forward to the next iterator, otherwise it moves forward without
// issue. This is based on the for_each STL function, but allows a reference to
// the second argument
template<class InputIterator, class Function, typename Arg>
Function binaryForEach(InputIterator First, InputIterator Last, Function F,
                       Arg &Second)
{
  for ( ; First!=Last; ++First ) {
    F(*First, Second);
  }
  return F;
}

template<class InputIterator, class Function, typename Arg>
Function safeBinaryForEach(InputIterator First, InputIterator Last, Function F,
                           Arg &Second)
{
  for ( ; First!=Last; ++First ) {
    if (F(*First, Second)) {
      --First;
    }
  }
  return F;
}

// A template function that has two levels of looping before calling the
// function with the passed in argument. See binaryForEach for further
// explanation
template<class InputIterator, class Function, typename Arg>
Function binaryNestedForEach(InputIterator First, InputIterator Last,
                             Function F, Arg &Second)
{
  for ( ; First != Last; ++First) {
    binaryForEach(First->begin(), First->end(), F, Second);
  }
  return F;
}
template<class InputIterator, class Function, typename Arg>
Function safeBinaryNestedForEach(InputIterator First, InputIterator Last,
                                 Function F, Arg &Second)
{
  for ( ; First != Last; ++First) {
    safeBinaryForEach(First->begin(), First->end(), F, Second);
  }
  return F;
}

// Unlike the STL, a pointer to the iterator itself is passed in with the 'safe'
// versions of these functions This allows the function to handle situations
// such as invalidated iterators
template<class InputIterator, class Function>
Function safeForEach(InputIterator First, InputIterator Last, Function F)
{
  for ( ; First!=Last; ++First )  F(&First)
    ; // Do nothing.
  return F;
}

// A template function that has two levels of looping before calling the
// function with a pointer to the current iterator. See binaryForEach for
// further explanation
template<class InputIterator, class SecondIterator, class Function>
Function safeNestedForEach(InputIterator First, InputIterator Last,
                              SecondIterator S, Function F)
{
  for ( ; First != Last; ++First) {
    SecondIterator sf, sl;
    for (sf = First->begin(), sl = First->end();
         sf != sl; )  {
      if (!F(&sf)) {
        ++sf;
      } 
    }
  }
  return F;
}
