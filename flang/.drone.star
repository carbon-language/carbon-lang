def clang(arch):
    return {
        "kind": "pipeline",
        "name": "%s-clang" % arch,
        "steps": [
            {
                "name": "test",
                "image": "fedora",
                "commands": [
                    "dnf -y install clang cmake ninja-build lld llvm-devel libcxx-devel diffutils",
                    "mkdir build && cd build",
                    'env CC=clang CXX=clang++ CXXFLAGS="-UNDEBUG -stdlib=libc++" LDFLAGS="-fuse-ld=lld" cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..',
                    "ninja -j12",
                    "ctest --output-on-failure -j24",
                ],
            },
        ],

    }

def gcc(arch):
    return {
            "kind": "pipeline",
            "name": "%s-gcc" % arch,
            "steps": [
                {
                    "name": "test",
                    "image": "fedora",
                    "commands": [
                        "dnf -y install cmake ninja-build g++ llvm-devel diffutils",
                        "mkdir build && cd build",
                        'env CC=gcc CXX=g++ CXXFLAGS="-UNDEBUG" LDFLAGS="-fuse-ld=gold" cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..',
                        "ninja -j12",
                        "ctest --output-on-failure -j24",
                    ],
                },
            ],

        }

def main(ctx):
    return [
        clang("amd64"),
        clang("arm64"),
        gcc("amd64"),
        gcc("arm64"),
    ]

