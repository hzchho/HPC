#!/bin/bash

# 设置源文件和目标文件的名称
SRC_MAIN="heated_plate_openmp.c"
SRC_LIB="parallel_fun.c"
OBJ_LIB="parallel_fun.o"
EXEC_FILE="heated_plate_openmp"
DLL_FILE="libfunction.so"

# 1. 编译 parallel_fun.c 为目标文件
echo "Compiling $SRC_LIB to $OBJ_LIB ..."
gcc -c $SRC_LIB -o $OBJ_LIB -lpthread

# 2. 使用 -shared 参数生成动态链接库 function.so
echo "Compiling $SRC_LIB to generate $DLL_FILE ..."
gcc -shared $SRC_LIB -o $DLL_FILE -lpthread

# 3. 编译 heated_plate_openmp.c 文件，生成目标文件，并链接 parallel_fun.o 和 function.so 生成最终的可执行文件
echo "Linking $SRC_MAIN and $OBJ_LIB to generate $EXEC_FILE ..."
gcc -g -Wall -fopenmp -o $EXEC_FILE $SRC_MAIN $OBJ_LIB -L. -lfunction -lpthread

# 4. 如果编译和链接成功，运行可执行文件
if [ $? -eq 0 ]; then
    echo "Compilation and linking succeeded, running $EXEC_FILE ..."
    valgrind --tool=massif --time-unit=B --stacks=yes ./$EXEC_FILE
else
    echo "Compilation or linking failed!"
fi
