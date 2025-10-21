#include <stdio.h>
#include <string.h>
#define MAXN 1000005
#define MAXE 4000005
#define MOD 100003

int head[MAXN], to[MAXE], nxt[MAXE], ec;
int dist[MAXN], ways[MAXN], q[MAXN];

void add(int u,int v){ to[++ec]=v; nxt[ec]=head[u]; head[u]=ec; }

int main() {
    int N,M,x,y; scanf("%d%d",&N,&M);
    while(M--){ scanf("%d%d",&x,&y); add(x,y); add(y,x); }

    memset(dist,-1,sizeof(dist));
    int l=0,r=0; q[r++]=1; dist[1]=0; ways[1]=1;

    while(l<r){
        int u=q[l++];
        for(int e=head[u]; e; e=nxt[e]){
            int v=to[e];
            if(dist[v]==-1){ dist[v]=dist[u]+1; ways[v]=ways[u]; q[r++]=v; }
            else if(dist[v]==dist[u]+1){ ways[v]+=ways[u]; if(ways[v]>=MOD) ways[v]-=MOD; }
        }
    }
    for(int i=1;i<=N;i++) printf("%d\n", ways[i]%MOD);
    return 0;
}
