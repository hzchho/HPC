import java.util.Scanner;
import java.text.DecimalFormat;

public class test1 {
    public static void main(String[] args){
        Scanner reader=new Scanner(System.in);
        int M=reader.nextInt();
        int N=reader.nextInt();
        int K=reader.nextInt();
        //分配空间
        double A[][]=new double[M][N];
        double B[][]=new double[N][K];
        double C[][]=new double[M][K];
        //初始化
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                double num1=Math.random();
                A[i][j]=num1;
            }
        }

        for(int i=0;i<N;i++){
            for(int j=0;j<K;j++){
                double num2=Math.random();
                B[i][j]=num2;
            }
        }

        for(int i=0;i<M;i++){
            for(int j=0;j<K;j++){
                C[i][j]=0.0;
            }
        }
        //开始时间
        double startTime=System.nanoTime();
        //矩阵运算
        for(int i=0;i<M;i++){
            for(int j=0;j<K;j++){
                for(int l=0;l<N;l++){
                    C[i][j]+=A[i][l]*B[l][j];
                }
            }
        }
        //结束时间
        double endTime=System.nanoTime();
        double Usedtime=endTime-startTime;

        DecimalFormat df=new DecimalFormat("#.######");
        System.out.println("Used time: "+df.format(Usedtime/1e9)+" s");
    }
}