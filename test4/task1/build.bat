@echo off
REM 设置源文件和目标文件的名称
set SRC_MAIN=heated_plate_openmp.c
set SRC_LIB=parallel_fun.c
set OBJ_LIB=parallel_fun.o
set EXEC_FILE=heated_plate_openmp.exe
set DLL_FILE=function.dll

REM 1. 编译 parallel_fun.c 为目标文件
echo Compiling %SRC_LIB% to %OBJ_LIB% ...
gcc -c %SRC_LIB%

REM 2. 使用 -shared 参数生成动态链接库 function.dll
echo Compile %SRC_LIB% generate %DLL_FILE% ...
gcc -shared -o %DLL_FILE% %SRC_LIB%

REM 3. 链接目标文件并生成最终可执行文件
echo Linking %OBJ_MAIN% and parallel_fun.o to generate %EXEC_FILE% ...
gcc -g -Wall -fopenmp -o %EXEC_FILE% %SRC_MAIN% -L. -lfunction

REM 4. 如果编译和链接成功，运行可执行文件
if %ERRORLEVEL% == 0 (
    echo Compilation and linking succeeded, running %EXEC_FILE% ...
    %EXEC_FILE%
) else (
    echo Compilation or linking failed!
)

pause
