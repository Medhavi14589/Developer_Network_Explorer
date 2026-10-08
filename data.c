#include<stdio.h>
#include<math.h>
#include<stdlib.h>

int main(){
  int n,sum=0;
  printf("Enter the number: \n");
  scanf("%d",&n);
  for(int i=0;i<n;i++){
    sum=sum+i;
  }
  printf("Sum: %d",sum);
  return 0;
}
