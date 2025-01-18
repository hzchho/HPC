@echo off
REM 设置MPI的路径
set PATH1=D:\MPI\MPI_SDK\Lib\x64\
set PATH2=D:\MPI\MPI_SDK\Include\

REM 设置源文件和目标文件的名称
set SRC_MAIN=heated_plate_openmp.c
set EXEC_FILE=heated_plate_openmp


REM 1. 链接目标文件并生成最终可执行文件
echo Linking %SRC_MAIN% and parallel_fun.o to generate %EXEC_FILE% ...
REM gcc -g -Wall -fopenmp -o %EXEC_FILE% %SRC_MAIN%
gcc %SRC_MAIN% -o %EXEC_FILE% -fopenmp -l msmpi -L %PATH1%  -I %PATH2% 

REM 2. 如果编译和链接成功，运行可执行文件
if %ERRORLEVEL% == 0 (
    echo Compilation and linking succeeded, running %EXEC_FILE% ...
    mpiexec -n 4 %EXEC_FILE%
    REM  %EXEC_FILE%
) else (
    echo Compilation or linking failed!
)

pause
