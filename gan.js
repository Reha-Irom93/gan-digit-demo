// Generator inference for the trained DCGAN and MLP-GAN (weights are float16, decoded to float32).
function f16(h){const s=(h&0x8000)?-1:1,e=(h>>10)&31,m=h&1023;if(e===0)return s*Math.pow(2,-14)*(m/1024);if(e===31)return m?NaN:s*Infinity;return s*Math.pow(2,e-15)*(1+m/1024);}
function decode(b64){const bin=atob(b64),n=bin.length>>1,o=new Float32Array(n);for(let i=0;i<n;i++)o[i]=f16(bin.charCodeAt(2*i)|(bin.charCodeAt(2*i+1)<<8));return o;}
function leaky(a){for(let i=0;i<a.length;i++)if(a[i]<0)a[i]*=0.2;return a;}
function dense(x,W,b,nin,nout){const y=new Float32Array(nout);if(b)y.set(b);for(let i=0;i<nin;i++){const xi=x[i],r=i*nout;for(let j=0;j<nout;j++)y[j]+=xi*W[r+j];}return y;}
function bn(x,g,b,m,v,C){const y=new Float32Array(x.length);for(let i=0;i<x.length;i++){const c=i%C;y[i]=(x[i]-m[c])/Math.sqrt(v[c]+1e-3)*g[c]+b[c];}return y;}
// transposed conv, stride 2, kernel 4, "same" padding; kernel layout (kh,kw,out,in); input (H,W,Cin)
function convT(x,H,W,Cin,K,Cout,bias){const OH=2*H,OW=2*W,y=new Float32Array(OH*OW*Cout);if(bias)for(let p=0;p<OH*OW;p++)for(let c=0;c<Cout;c++)y[p*Cout+c]=bias[c];
 for(let i=0;i<H;i++)for(let j=0;j<W;j++)for(let kh=0;kh<4;kh++){const oy=2*i+kh-1;if(oy<0||oy>=OH)continue;for(let kw=0;kw<4;kw++){const ox=2*j+kw-1;if(ox<0||ox>=OW)continue;
  const xo=(i*W+j)*Cin,yo=(oy*OW+ox)*Cout,ko=(kh*4+kw)*Cout*Cin;for(let co=0;co<Cout;co++){let s=0;const kk=ko+co*Cin;for(let ci=0;ci<Cin;ci++)s+=x[xo+ci]*K[kk+ci];y[yo+co]+=s;}}}
 return y;}
function genDC(P,z){let h=dense(z,P[0],null,100,6272);h=leaky(bn(h,P[1],P[2],P[3],P[4],128));h=leaky(bn(convT(h,7,7,128,P[5],64,null),P[6],P[7],P[8],P[9],64));h=convT(h,14,14,64,P[10],1,P[11]);return h.map(Math.tanh);}
function genMLP(P,z){let h=leaky(dense(z,P[0],P[1],100,256));h=leaky(dense(h,P[2],P[3],256,512));h=dense(h,P[4],P[5],512,784);return h.map(Math.tanh);}
