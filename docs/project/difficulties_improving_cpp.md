# Difficulties improving C++

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

C++ is the dominant programming language for the performance critical software
our goals prioritize. The most direct way to deliver a modern and excellent
developer experience for those use cases and developers would be to improve C++.

Improving C++ to deliver the kind of experience developers expect from a
programming language today is difficult in part because **C++ has decades of
technical debt** accumulated in the design of the language. It inherited the
legacy of C, including
[textual preprocessing and inclusion](https://clang.llvm.org/docs/Modules.html#problems-with-the-current-model).
At the time, this was essential to C++'s success by giving it instant and high
quality access to a large C ecosystem. However, over time this has resulted in
significant technical debt ranging from
[integer promotion rules](https://shafik.github.io/c++/2021/12/30/usual_arithmetic_confusions.html)
to complex syntax with
"[the most vexing parse](https://en.wikipedia.org/wiki/Most_vexing_parse)".

**C++ has also prioritized backwards compatibility** including both syntax and
[ABI](https://en.wikipedia.org/wiki/Application_binary_interface). This is
heavily motivated by preserving its access to existing C/C++ ecosystems, and
forms one of the foundations of common Linux package management approaches. A
consequence is that rather than changing or replacing language designs to
simplify and improve the language, features have overwhelmingly been added over
time. This both creates technical debt due to complicated feature interaction,
and fails to benefit from on cleanup opportunities in the form of replacing or
removing legacy features.

Carbon is exploring significant backwards incompatible changes. It doesn't
inherit the legacy of C or C++ directly, and instead is starting with solid
foundations, like a modern generics system, modular code organization, and
consistent, simple syntax. Then, it builds a simplified and improved language
around those foundational components that remains both interoperable with and
migratable from C++, while giving up transparent backwards compatibility. This
is fundamentally **a successor language approach**, rather than an attempt to
incrementally evolve C++ to achieve these improvements.

Another challenge to improving C++ in these ways is the current evolution
process and direction. A key example of this is the committee's struggle to
converge on a clear set of high-level and long-term goals and priorities aligned
with [ours](https://wg21.link/p2137). When [pushed](https://wg21.link/p1863) to
address
[the technical debt caused by not breaking the ABI](https://wg21.link/p2028),
**C++'s process
[did not reach any definitive conclusion](https://cor3ntin.github.io/posts/abi/#abi-discussions-in-prague)**.
This both failed to meaningfully change C++'s direction and priorities towards
improvements rather than backwards compatibility, and demonstrates how the
process can fail to make directional decisions.

Beyond C++'s evolution direction, the mechanics of the process also make
improving C++ difficult. **C++'s process is oriented around standardization
rather than design**: it uses a multiyear waterfall committee process. Access to
the committee and standard is restricted and expensive, attendance is necessary
to have a voice, and decisions are made by live votes of those present. The
committee structure is designed to ensure representation of nations and
companies, rather than building an inclusive and welcoming team and community of
experts and people actively contributing to the language.

Carbon has a more accessible and efficient [evolution process](evolution.md)
built on open-source principles, processes, and tools. Throughout the project,
we explicitly and clearly lay out our [goals and priorities](goals.md) and how
those directly shape our decisions. We also have a clear
[governance structure](evolution.md#governance-structure) that can make
decisions rapidly when needed. The open-source model enables the Carbon project
to expand its scope beyond just the language. We will build a holistic
collection of tools that provide a rich developer experience, ranging from the
compiler and standard library to IDE tools and more. **We will even try to close
a huge gap in the C++ ecosystem with a built-in package manager.**

Carbon is particularly focused on a specific set of [goals](goals.md). These
will not align with every user of C++, but have significant interest across a
wide range of users that are capable and motivated to evolve and modernize their
codebase. Given the difficulties posed by C++'s technical debt, sustained
priority of backwards compatibility, and evolution process, we wanted to explore
an alternative approach to achieve these goals -- through a
backwards-incompatible successor language, designed with robust support for
interoperability with and migration from C++. We hope other efforts to
incrementally improve C++ continue, and would love to share ideas where we can.

--------------- SPANISH TRANSLATION ---------------

Dificultades para mejorar C++

C++ es el lenguaje de programación por excelencia para la mejorar el rendimiento 
de software crítico que priorizamos en nuestros objetivos. La forma más directa de
ofrecer una moderna y excelente experiencia de desarrollador para esos casos de uso 
sería mejorar C++.

Mejorar C++ para brindar el tipo de experiencia que los desarrolladores esperan de 
unlenguaje de programación actual es difícil, en parte porque C++ tiene décadas de 
revisiones técnicas acumulada en el diseño del lenguaje. Heredó el legado de C, 
incluido el preprocesamiento y la inclusión textual. En ese momento, esto era esencial
para el éxito de C++ al brindarle acceso instantáneo y de alta calidad a un gran 
ecosistema de C. Sin embargo, con el tiempo esto ha resultado en una pesada herencia
técnica significativa que va desde reglas de integer promotion hasta sintaxis compleja
con "el análisis más desconcertante (vexing parse)".

C ++ también ha priorizado la compatibilidad con versiones anteriores, incluída la 
sintaxis y ABI. Esto está fuertemente motivado por la preservación de su acceso a 
los ecosistemas C/C++ existentes y forma una de las bases de los enfoques comunes de
administración de paquetes de Linux. Una consecuencia es que, en lugar de cambiar o
reemplazar los diseños de lenguaje para simplificar y mejorar el lenguaje, se han 
agregado características abrumadoras con el tiempo. Esto crea una deuda técnica 
debido a la complicada interacción de funciones y no se beneficia de las oportunidades 
de limpieza en forma de reemplazo o eliminación de funciones heredadas.

Carbon está explorando cambios significativos incompatibles con versiones anteriores. 
No hereda el legado de C o C++ directamente, sino que comienza con bases sólidas, 
como un sistema genérico moderno, una organización de código modular y una sintaxis 
simple y consistente. Luego, crea un lenguaje simplificado y mejorado en torno a esos
componentes fundamentales que siguen siendo interoperables y migrables desde C++, al
tiempo que renuncian a la compatibilidad con versiones anteriores transparentes. 
Este es fundamentalmente un enfoque de lenguaje sucesor, en lugar de un intento de 
evolucionar gradualmente C++ para lograr estas mejoras.

Otro desafío para mejorar C++ de esta manera es la dirección y el proceso de evolución actual. 
Un ejemplo clave de esto es la lucha del comité para converger en un conjunto claro de objetivos 
y prioridades de alto nivel y largo plazo alineados con los nuestros. Cuando se le presionó para
abordar la deuda técnica causada por no romper la ABI, el proceso de C++ no llegó a ninguna 
conclusión definitiva. Esto no logró cambiar significativamente la dirección y las prioridades de 
C ++ hacia las mejoras en lugar de la compatibilidad con versiones anteriores, y demuestra cómo el 
proceso puede fallar al tomar decisiones direccionales.

Más allá de la dirección de evolución de C++, la mecánica del proceso también dificulta la mejora 
de C++. El proceso de C++ está orientado a la estandarización más que al diseño: utiliza una 
estructura de comité en cascada de hace varios años. El acceso al comité y a la norma es restringido
y costoso, la asistencia es necesaria para tener voz y las decisiones se toman por votos en vivo de
los presentes. La estructura del comité está diseñada para garantizar la representación de naciones 
y empresas, en lugar de crear un equipo y una comunidad más inclusivos y acogedores de expertos y 
personas que contribuyan activamente al idioma.

Carbon tiene un proceso de evolución más accesible y eficiente basado en principios, procesos y 
herramientas de código abierto. A lo largo del proyecto, establecemos explícita y claramente 
nuestros objetivos y prioridades y cómo estos dan forma directamente a nuestras decisiones. 
También contamos con una estructura de gobierno clara que puede tomar decisiones rápidamente cuando 
sea necesario. El modelo de código abierto permite que el proyecto Carbon amplíe su alcance más 
allá del lenguaje. Construiremos una colección holística de herramientas que brinden una rica 
experiencia de desarrollador, desde el compilador y la biblioteca estándar hasta las herramientas 
IDE y mucho más. Incluso intentaremos cerrar una gran brecha en el ecosistema de C++ con un 
administrador de paquetes integrado.

El lenguaje Carbon se centra especialmente en un conjunto específico de objetivos. Estos no se 
alinearán con todos los usuarios de C ++, pero tienen un interés significativo en una amplia gama 
de usuarios que son capaces y están motivados para evolucionar y modernizar su código de base.
Dadas las dificultades planteadas por la deuda técnica de C++, la prioridad sostenida de la 
compatibilidad con versiones anteriores y el proceso de evolución, queríamos explorar un enfoque 
alternativo para lograr estos objetivos, a través de un lenguaje sucesor incompatible con versiones 
anteriores, diseñado con soporte sólido para la interoperabilidad y la migración desde C++. 
Esperamos que continúen nuestros esfuerzos y los de otros para mejorar gradualmente C++ y nos encantaría 
compartir ideas allí donde podamos.
