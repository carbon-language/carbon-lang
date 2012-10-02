// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s | FileCheck %s
// rdar://12378879

/**
 * \param[in] arg1 ZZZ
 * \param[out] d xxx
*/
void cfunction_availability(int arg1, double d) __attribute__((availability(macosx,obsoleted=10.0,introduced=8.0,deprecated=9.0, message="use availability_test")))
                                                __attribute__((availability(ios,unavailable, message="not for iOS")));


// CHECK: annotate-comments-availability-attrs.cpp:8:6: FunctionDecl=cfunction_availability:{{.*}} FullCommentAsXML=[<Function file="{{.*}}annotate-comments-availability-attrs.cpp" line="8" column="6"><Name>cfunction_availability</Name><USR>c:@F@cfunction_availability#I#d#</USR><Parameters><Parameter><Name>arg1</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter><Parameter><Name>d</Name><Index>1</Index><Direction isExplicit="1">out</Direction><Discussion><Para> xxx</Para></Discussion></Parameter></Parameters><Availability distribution="iOS"> <DeprecationSummary>not for iOS</DeprecationSummary><Unavailable>true</Unavailable></Availability><Availability distribution="OS X"><IntroducedInVersion>8.0</IntroducedInVersion><DeprecatedInVersion>9.0</DeprecatedInVersion><RemovedAfterVersion>10.0</RemovedAfterVersion> <DeprecationSummary>use availability_test</DeprecationSummary></Availability></Function>]


/**
 * \param[in] arg1 ZZZ
 * \param[out] d xxx
 */
void dep(int arg1, double d) __attribute__((deprecated));

// CHECK: annotate-comments-availability-attrs.cpp:19:6: FunctionDecl=dep:{{.*}} FullCommentAsXML=[<Function file="{{.*}}annotate-comments-availability-attrs.cpp" line="19" column="6"><Name>dep</Name><USR>c:@F@dep#I#d#</USR><Parameters><Parameter><Name>arg1</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter><Parameter><Name>d</Name><Index>1</Index><Direction isExplicit="1">out</Direction><Discussion><Para> xxx </Para></Discussion></Parameter></Parameters><Deprecated>true</Deprecated></Function>


/**
 * \param[in] arg1 ZZZ
 */
void unv(int arg1) __attribute__((unavailable));

// CHECK: annotate-comments-availability-attrs.cpp:27:6: FunctionDecl=unv:{{.*}} FullCommentAsXML=[<Function file="{{.*}}annotate-comments-availability-attrs.cpp" line="27" column="6"><Name>unv</Name><USR>c:@F@unv#I#</USR><Parameters><Parameter><Name>arg1</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter></Parameters><Unavailable>true</Unavailable></Function>
