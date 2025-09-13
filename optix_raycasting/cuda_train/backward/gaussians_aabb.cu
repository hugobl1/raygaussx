#include <optix.h>

#include "vec_math.h"
#include "helpers.h"
#include "gaussians_aabb.h"

__forceinline__ __device__
void quaternion_to_matrix(const float4& q, float3& c0, float3& c1, float3& c2){
    const float r=q.x,i=q.y,j=q.z,k=q.w;
    const float ii=i+i, jj=j+j, kk=k+k;
    const float ri=r*ii, rj=r*jj, rk=r*kk;
    const float ii2=i*ii, jj2=j*jj, kk2=k*kk;
    const float ij=i*jj, ik=i*kk, jk=j*kk;
    c0=make_float3(1.f-(jj2+kk2), ij+rk,        ik-rj);
    c1=make_float3(ij-rk,         1.f-(ii2+kk2), jk+ri);
    c2=make_float3(ik+rj,         jk-ri,        1.f-(ii2+jj2));
}

__forceinline__ __device__
void apply_Sinv_Rt(const float3& x,
                   const float3& U, const float3& V, const float3& W, // colonnes de R
                   const float3& inv_s,
                   float3& out)
{
    // r = R^T x
    const float rx = fmaf(U.x,x.x, fmaf(U.y,x.y, U.z*x.z));
    const float ry = fmaf(V.x,x.x, fmaf(V.y,x.y, V.z*x.z));
    const float rz = fmaf(W.x,x.x, fmaf(W.y,x.y, W.z*x.z));
    out = make_float3(rx*inv_s.x, ry*inv_s.y, rz*inv_s.z);
}


extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __intersection__gaussian()
{
    const unsigned int primitive_index = optixGetPrimitiveIndex();
    const float3 ray_direction  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float4 invI4 = __ldg(&params.prims[primitive_index].inv_inter4);
    const float4 quaternion    = __ldg(&params.prims[primitive_index].quat4);
    const float4 p4    = __ldg(&params.prims[primitive_index].pos4);

    const float3 inv_scales=make_float3(invI4);
    const float3 O=make_float3(p4);

    float3 U_rot,V_rot,W_rot;
    quaternion_to_matrix(quaternion,U_rot,V_rot,W_rot);

    float3 O_ellipsis, dir_ellipsis;
    apply_Sinv_Rt(O, U_rot, V_rot, W_rot, inv_scales, O_ellipsis);
    apply_Sinv_Rt(ray_direction, U_rot, V_rot, W_rot, inv_scales, dir_ellipsis);
    
    const float a=fmaf(dir_ellipsis.x, dir_ellipsis.x,
               fmaf(dir_ellipsis.y, dir_ellipsis.y, dir_ellipsis.z*dir_ellipsis.z));
    float b = -fmaf(O_ellipsis.x, dir_ellipsis.x,
               fmaf(O_ellipsis.y, dir_ellipsis.y,
                    O_ellipsis.z * dir_ellipsis.z));

    float c = fmaf(O_ellipsis.x, O_ellipsis.x,
               fmaf(O_ellipsis.y, O_ellipsis.y,
                    fmaf(O_ellipsis.z, O_ellipsis.z, -1.0f)));

    const float inv_a = __fdividef(1.0f, a);
    const float b_inv_a    = b * inv_a;

    const float3 P = make_float3(
        fmaf(b_inv_a, dir_ellipsis.x, O_ellipsis.x),
        fmaf(b_inv_a, dir_ellipsis.y, O_ellipsis.y),
        fmaf(b_inv_a, dir_ellipsis.z, O_ellipsis.z));
    const float d2   = fmaf(P.x, P.x, fmaf(P.y, P.y, P.z*P.z));
    const float disc = 1.0f - d2;

    if( disc > 1e-7f )
    {
        float sdisc        = sqrtf( disc*a );
        float q = b + copysignf(sdisc, b);
        float root1        = (c/q);
        float root2        = q*inv_a;

        float min_t= fmaxf(ray_tmin,root1);
        float max_t= fminf(ray_tmax,root2);

        if ((min_t<=max_t)){
            optixReportIntersection( min_t,0);
        }

    }
}



static __forceinline__ __device__ void computeRay( const uint3 idx, float3& origin, float3& direction)
{  
    const float idx_x = (idx.x+0.5f)/params.image_width;
    const float idx_y = (idx.y+0.5f)/params.image_height;

    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 cam_tan_half_fov=make_float2(params.cam_intr.x,params.cam_intr.y);
    const float2 pixel_idx = 2.0f*(make_float2(idx_x-params.cam_intr.z,idx_y-params.cam_intr.w));

    const float2 d = pixel_idx * cam_tan_half_fov;
    origin    = params.cam_eye;
    direction = normalize( d.x * U + d.y * V + W );
}

template<int CS>
static __forceinline__ __device__
void accumulateChunk_backw(const unsigned int idx_ray, const unsigned int p0,
                     const float tbuffer, const float3 ray_direction, const float dt,
                     const int i_begin, float3 diff_color, float2 (&buf)[CS])
{
    // zero local chunk buffer
    #pragma unroll
    for (int i=0;i<CS;++i) buf[i] = make_float2(0.f);

    // const float  t_mid         = tbuffer + 0.5f * dt;      // base pour i=0
    const float  t_mid         = fmaf(dt, (float)i_begin + 0.5f, tbuffer);
    const float3 ray_dir_tbuff = ray_direction * t_mid;

    int pid = params.hit_prim_idx[idx_ray * params.max_prim_slice + 0];
    float4 pos4_next  = __ldg(&params.prims[pid].pos4);
    float4 invI4_next = __ldg(&params.prims[pid].inv_inter4);
    float4 quat_next  = __ldg(&params.prims[pid].quat4);
    float4 rgba_next  = __ldg(&params.prims[pid].rgba4);

    #pragma unroll 1   // evite de gonfler les registres
    for (int prim_iter = 0; prim_iter < (int)p0; ++prim_iter)
    {
        const float3 gaussian_pos_rel = make_float3(pos4_next);
        const float  k                = invI4_next.w;
        const float3 inv_scales       = make_float3(invI4_next) * k;
        const float  power_cut        = k * k;
        const float4 quaternion       = quat_next;
        const float4 rgba             = rgba_next;
        const float sigma=rgba.w;
        const float dot_diff_color_gauss_color=dot(diff_color,make_float3(rgba));

        if (prim_iter + 1 < (int)p0) {
            const int pid2 = params.hit_prim_idx[idx_ray * params.max_prim_slice + (prim_iter + 1)];
            pos4_next  = __ldg(&params.prims[pid2].pos4);
            invI4_next = __ldg(&params.prims[pid2].inv_inter4);
            quat_next  = __ldg(&params.prims[pid2].quat4);
            rgba_next  = __ldg(&params.prims[pid2].rgba4);
        }

        float3 U_rot, V_rot, W_rot;
        quaternion_to_matrix(quaternion, U_rot, V_rot, W_rot);

        const float3 xhit_xgauss = gaussian_pos_rel + ray_dir_tbuff;

        float3 Mx0, Md;
        apply_Sinv_Rt(xhit_xgauss,  U_rot, V_rot, W_rot, inv_scales, Mx0);
        apply_Sinv_Rt(ray_direction, U_rot, V_rot, W_rot, inv_scales, Md);

        const float Md2       = dot(Md, Md);
        const float Md_dt_2   = (dt * dt) * Md2;

        float power = dot(Mx0, Mx0);
        float inc   = Md_dt_2 + 2.0f * dot(Mx0, Md) * dt;
        const float inc_st = 2.0f * Md_dt_2;

        #pragma unroll
        for (int i = 0; i < CS; ++i) {
            if (power < power_cut) {
                const float gw = __expf(-0.5f * power);
                float2 &b = buf[i];
                b.x += rgba.w * gw;
                b.y += dot_diff_color_gauss_color * gw;
                // b.y += rgba.x * gw;
                // b.z += rgba.y * gw;
                // b.w += rgba.z * gw;
            }
            power += inc;
            inc   += inc_st;
        }
    }
}

// template<int BS, int CS>
// static __forceinline__ __device__
// void colorBlendingT_chunked(const unsigned int idx_ray, const unsigned int p0,
//                             const float tbuffer, const float3 ray_direction, const float dt,
//                             float3 &ray_color, float &transmittance)
// {

//     #pragma unroll 1
//     for (int base = 0; base < BS && transmittance > TRANSMITTANCE_EPSILON; base += CS) {
//         float4 local[CS];
//         accumulateChunk<CS>(idx_ray, p0, tbuffer, ray_direction, dt, base, local);

//         #pragma unroll
//         for (int i = 0; i < CS; ++i) {
//             const float x = local[i].x * dt;
//             const float w = __expf(-x);
//             const float s = dt * __fdividef(1.0f - w, fmaxf(x, 1e-6f));
//             ray_color     += transmittance * s * make_float3(local[i].y, local[i].z, local[i].w);
//             transmittance *= w;
//         }
//     }
// }

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx_ray= optixGetLaunchIndex();
    const unsigned int idx_ray_flatten = idx_ray.y * params.image_width + idx_ray.x;

    //Les gradients sont accumules dans 4 buffers intermediaires avant d'etre ecrits en memoire globale, l'indice du buffer pour un rayon donne est son indice modulo 4
    const int gradient_buffer_idx= idx_ray_flatten % 4;
    float3 ray_origin, ray_direction;
    computeRay( idx_ray, ray_origin, ray_direction);

    const float3 bbox_min = params.bbox_min;
    const float3 bbox_max = params.bbox_max;

    float3 t0,t1,tmin,tmax;
    t0 = (bbox_min - ray_origin) / ray_direction;
    t1 = (bbox_max - ray_origin) / ray_direction;
    tmin = fminf(t0, t1);
    tmax = fmaxf(t0, t1);
    float tenter=fmaxf(0.0f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z)));
    float texit=fminf(tmax.x, fminf(tmax.y, tmax.z));

    const float dt_base = params.dt_step;
    float dt = dt_base; // Initial step size
    float slab_spacing = dt*BUFFER_SIZE;
 
    unsigned  int p0;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        tenter,
        texit,
        0.0f,                // rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                   // SBT offset
        2,                   // SBT stride
        0,                   // missSBTIndex
        p0
        );
    if (__int_as_float(p0)==-1.0f){
        return;
    }
    tenter=__int_as_float(p0);

    if(tenter<texit){
        float tbuffer=tenter;
        float t_min_slab;
        float t_max_slab;
        unsigned int p0=0;

        float transmittance_backward=1.0f;
        float3 ray_color_backward = params.ray_colors[idx_ray_flatten];

        float3 diff_color=params.dloss_dray_colors[idx_ray_flatten];

        float dot_diff_col_ray_col_back = dot(diff_color, ray_color_backward);
        
        while(tbuffer<texit && transmittance_backward>TRANSMITTANCE_EPSILON){
            if (params.dynamic_sampling){
                float dt_dist=fmaxf(tbuffer/1024.0f,dt_base);
                float coeff_transmittance = rcbrtf(transmittance_backward);
                dt=fminf(dt_dist*coeff_transmittance,4.0f*dt_base);
                slab_spacing=dt*BUFFER_SIZE;
            }

        p0=0;

        // t_min_slab = fmaxf(tenter,tbuffer);
        // t_max_slab = fminf(texit, tbuffer + slab_spacing);
        t_min_slab = tbuffer;
        t_max_slab = tbuffer + slab_spacing;
        if(t_max_slab>tenter)
        {

        optixTrace(
                params.handle,
                ray_origin,
                ray_direction,
                t_min_slab,
                t_max_slab,
                0.0f,                // rayTime
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                1,                   // SBT offset
                2,                   // SBT stride
                1,                   // missSBTIndex
                p0
                );

        if(p0==0){
            tbuffer+=slab_spacing;
            unsigned  int next_tbuffer;
            optixTrace(
                params.handle,
                ray_origin,
                ray_direction,
                tbuffer,
                texit,
                0.0f,                // rayTime
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0,                   // SBT offset
                2,                   // SBT stride
                0,                   // missSBTIndex
                next_tbuffer
                );
            if (__int_as_float(next_tbuffer)==-1.0f){
                break;
            }
            tbuffer=__int_as_float(next_tbuffer);
            continue;
        }

        const int NB = BUFFER_SIZE / CHUNK_SIZE; // ici 2
        float2 buffers[NB][CHUNK_SIZE];
        for (int b = 0; b < NB; ++b){
            accumulateChunk_backw<CHUNK_SIZE>(idx_ray_flatten, p0, tbuffer, ray_direction, dt, b*CHUNK_SIZE,diff_color, buffers[b]);
        }

        //Petite boucle calculant transmittance, dot(diff_color,current_color_ray)  et dot(diff_color,current_color)
        float transmittance_aft_buffer=transmittance_backward;
        float dot_diff_col_ray_col_back_after=dot_diff_col_ray_col_back;
        for (int b = 0; b < NB; ++b){
            float2* temp_buff=buffers[b];
            for(int index_buffer=0; index_buffer<CHUNK_SIZE; index_buffer++){
                const float sample_dens= temp_buff[index_buffer].x;
                // float3 sample_col= make_float3(temp_buff[index_buffer].y, temp_buff[index_buffer].z, temp_buff[index_buffer].w);
                float dot_diff_color_cur_col= temp_buff[index_buffer].y;
                float inv_density=(sample_dens>0.0f) ? __fdividef(1.0f, sample_dens) : 0.0f;
                // sample_col*=inv_density;
                dot_diff_color_cur_col *= inv_density;
                const float w=__expf(-sample_dens*dt);
                // float dot_diff_color_cur_col = dot(diff_color, sample_col);

                float alpha_T_div_sig=transmittance_aft_buffer*(1.0f-w)*inv_density;
                // buffers[b][index_buffer].x=alpha_T_div_sig;
                buffers[b][index_buffer].y=dot_diff_color_cur_col;
                // buffers[b][index_buffer].z=transmittance_aft_buffer;
                // buffers[b][index_buffer].w=dot_diff_col_ray_col_back_after;
                dot_diff_col_ray_col_back_after -= transmittance_aft_buffer*(1.0f-w)*dot_diff_color_cur_col;
                transmittance_aft_buffer*=w;
            }
        }
        for(int prim_iter=0;prim_iter<p0; prim_iter++){
            float dot_diff_col_ray_col_back_temp=dot_diff_col_ray_col_back;
            float transmittance_temp=transmittance_backward;
            float grad_color = 0.0f;
            float dloss_dsigma=0.0f;
            float3 dloss_dscale=make_float3(0.0f);
            float3 dloss_dpos=make_float3(0.0f);

            float3 dloss_dr123_1=make_float3(0.0f);
            float3 dloss_dr123_2=make_float3(0.0f);
            float3 dloss_dr123_3=make_float3(0.0f);

            int primitive_index= params.hit_prim_idx[idx_ray_flatten * params.max_prim_slice + prim_iter];
            PrimData p=params.prims[primitive_index];
            const float3 gaussian_pos_rel=make_float3(p.pos4);
            const float  k                = p.inv_inter4.w;
            const float3 inv_scales=make_float3(p.inv_inter4)*k;
            const float  power_cut        = k * k;
            float4 quaternion=p.quat4;
            float3 U_rot,V_rot,W_rot;
            quaternion_to_matrix(quaternion,U_rot,V_rot,W_rot);
            //M=(M1,M2,M3) = (RS^{-1})^T where S is the scaling matrix and R the rotation matrix so Sigma^{-1}=M^T*M
            float3 M1,M2,M3;
            U_rot=U_rot*inv_scales.x;
            V_rot=V_rot*inv_scales.y;
            W_rot=W_rot*inv_scales.z;
            M1=make_float3(U_rot.x,V_rot.x,W_rot.x);
            M2=make_float3(U_rot.y,V_rot.y,W_rot.y);
            M3=make_float3(U_rot.z,V_rot.z,W_rot.z);

            // float3 gaussian_color = make_float3(params.color_features[primitive_index*3],params.color_features[primitive_index*3+1],params.color_features[primitive_index*3+2]);
            float gaussian_density=p.rgba4.w;
            const float dot_diff_color_gauss_col=dot(make_float3(p.rgba4),diff_color)/p.rgba4.w;


            float t0=tbuffer+0.5f*dt;
            float3 xhit_xgaus=gaussian_pos_rel+ray_direction*t0;
            float3 M_xhit_xgaus=xhit_xgaus.x*M1+xhit_xgaus.y*M2+xhit_xgaus.z*M3;
            const float3 Md= ray_direction.x*M1+ray_direction.y*M2+ray_direction.z*M3;
            const float Md2=dot(Md,Md);
            const float Md_dt_2=(dt*dt)*Md2;
            float power=dot(M_xhit_xgaus,M_xhit_xgaus);
            float inc=Md_dt_2+2.0f*dot(M_xhit_xgaus,Md)*dt;
            const float inc_st=2.0f*Md_dt_2;

            for (int base = 0; base < BUFFER_SIZE && transmittance_temp > TRANSMITTANCE_EPSILON; base += CHUNK_SIZE) {
            int b=base/CHUNK_SIZE;
            float2 *buffer=buffers[b];
            for (int index_buffer = 0; index_buffer < CHUNK_SIZE; ++index_buffer) {
                const float sigma=buffer[index_buffer].x;
                const float w= __expf(-sigma*dt);
                float dot_diff_color_cur_col=buffers[b][index_buffer].y;
                if (power < power_cut) {
                    // const float alpha_T_div_sig=buffer[index_buffer].x;
                    const float alpha_T_div_sig=transmittance_temp*(1.0f-w)*((sigma>0.0f) ? __fdividef(1.0f, sigma) : 0.0f);
                    // float transmittance=buffers[b][index_buffer].z;
                    // float dot_diff_color_cur_col_ray=buffers[b][index_buffer].w;

                    float weight_density=__expf(-0.5f * power);
                    float sigma_weight=weight_density*gaussian_density;
                    
                    grad_color+=sigma_weight*alpha_T_div_sig;
                    float dloss_dsigma_aux=(transmittance_temp*dot_diff_color_cur_col-dot_diff_col_ray_col_back_temp)*dt;
                    dloss_dsigma+=dloss_dsigma_aux*weight_density;
                    
                    float dloss_dweights=dloss_dsigma_aux*gaussian_density;

                    dloss_dweights+=alpha_T_div_sig*(gaussian_density)*(dot_diff_color_gauss_col-dot_diff_color_cur_col);
                    dloss_dsigma+=alpha_T_div_sig*(weight_density)*(dot_diff_color_gauss_col-dot_diff_color_cur_col);

                    dloss_dscale.x+=dloss_dweights*weight_density*(M_xhit_xgaus.x*M_xhit_xgaus.x)*inv_scales.x ;
                    dloss_dscale.y+=dloss_dweights*weight_density*(M_xhit_xgaus.y*M_xhit_xgaus.y)*inv_scales.y ;
                    dloss_dscale.z+=dloss_dweights*weight_density*(M_xhit_xgaus.z*M_xhit_xgaus.z)*inv_scales.z ;

                    dloss_dpos.x+=dloss_dweights*weight_density*dot(M1, M_xhit_xgaus) ;
                    dloss_dpos.y+=dloss_dweights*weight_density*dot(M2, M_xhit_xgaus) ;
                    dloss_dpos.z+=dloss_dweights*weight_density*dot(M3, M_xhit_xgaus) ;

                    dloss_dr123_1-=dloss_dweights*weight_density*xhit_xgaus*(M_xhit_xgaus.x*inv_scales.x) ;
                    dloss_dr123_2-=dloss_dweights*weight_density*xhit_xgaus*(M_xhit_xgaus.y*inv_scales.y) ;
                    dloss_dr123_3-=dloss_dweights*weight_density*xhit_xgaus*(M_xhit_xgaus.z*inv_scales.z) ;
                }
                xhit_xgaus += ray_direction * dt;
                M_xhit_xgaus += Md * dt;
                power += inc;
                inc   += inc_st;
                dot_diff_col_ray_col_back_temp-=transmittance_temp*(1.0f-w)*dot_diff_color_cur_col;
                transmittance_temp *= w;
            }
        }
            //Remplacer atomicAdd par un stockage dans un buffer intermediaire
            unsigned int offset_index= primitive_index*4+gradient_buffer_idx;
            atomicAdd(&(params.color_features_grad[offset_index].x),diff_color.x*grad_color);
            atomicAdd(&(params.color_features_grad[offset_index].y),diff_color.y*grad_color);
            atomicAdd(&(params.color_features_grad[offset_index].z),diff_color.z*grad_color);
            // atomicAdd(&(params.color_features_grad[primitive_index].x),diff_color.x*grad_color);
            // atomicAdd(&(params.color_features_grad[primitive_index].y),diff_color.y*grad_color);
            // atomicAdd(&(params.color_features_grad[primitive_index].z),diff_color.z*grad_color);
            
            atomicAdd(&params.densities_grad[offset_index], dloss_dsigma);
            // atomicAdd(&params.densities_grad[primitive_index], dloss_dsigma);

            atomicAdd(&(params.scales_grad[offset_index].x),dloss_dscale.x);
            atomicAdd(&(params.scales_grad[offset_index].y),dloss_dscale.y);
            atomicAdd(&(params.scales_grad[offset_index].z),dloss_dscale.z);
            // atomicAdd(&(params.scales_grad[primitive_index].x),dloss_dscale.x);
            // atomicAdd(&(params.scales_grad[primitive_index].y),dloss_dscale.y);
            // atomicAdd(&(params.scales_grad[primitive_index].z),dloss_dscale.z);

            atomicAdd(&(params.positions_grad[offset_index].x),dloss_dpos.x);
            atomicAdd(&(params.positions_grad[offset_index].y),dloss_dpos.y);
            atomicAdd(&(params.positions_grad[offset_index].z),dloss_dpos.z);
            // atomicAdd(&(params.positions_grad[primitive_index].x),dloss_dpos.x);
            // atomicAdd(&(params.positions_grad[primitive_index].y),dloss_dpos.y);
            // atomicAdd(&(params.positions_grad[primitive_index].z),dloss_dpos.z);

            float4 quaternions_grad=make_float4(0.0f,0.0f,0.0f,0.0f);
            quaternions_grad.x=(2*quaternion.w*dloss_dr123_1.y-2*quaternion.z*dloss_dr123_1.z
                                        -2*quaternion.w*dloss_dr123_2.x+2*quaternion.y*dloss_dr123_2.z
                                        +2*quaternion.z*dloss_dr123_3.x-2*quaternion.y*dloss_dr123_3.y);
            quaternions_grad.y=(2*quaternion.z*dloss_dr123_1.y+2*quaternion.w*dloss_dr123_1.z
                                        +2*quaternion.z*dloss_dr123_2.x-4*quaternion.y*dloss_dr123_2.y+2*quaternion.x*dloss_dr123_2.z
                                        +2*quaternion.w*dloss_dr123_3.x-2*quaternion.x*dloss_dr123_3.y-4*quaternion.y*dloss_dr123_3.z);
            quaternions_grad.z=(-4*quaternion.z*dloss_dr123_1.x+2*quaternion.y*dloss_dr123_1.y-2*quaternion.x*dloss_dr123_1.z
                                        +2*quaternion.y*dloss_dr123_2.x+2*quaternion.w*dloss_dr123_2.z
                                        +2*quaternion.x*dloss_dr123_3.x+2*quaternion.w*dloss_dr123_3.y-4*quaternion.z*dloss_dr123_3.z);
            quaternions_grad.w=(-4*quaternion.w*dloss_dr123_1.x +2*quaternion.x*dloss_dr123_1.y+2*quaternion.y*dloss_dr123_1.z
                                        -2*quaternion.x*dloss_dr123_2.x-4*quaternion.w*dloss_dr123_2.y+2*quaternion.z*dloss_dr123_2.z
                                        +2*quaternion.y*dloss_dr123_3.x+2*quaternion.z*dloss_dr123_3.y);
            atomicAdd(&(params.quaternions_grad[offset_index].x),quaternions_grad.x);
            atomicAdd(&(params.quaternions_grad[offset_index].y),quaternions_grad.y);
            atomicAdd(&(params.quaternions_grad[offset_index].z),quaternions_grad.z);
            atomicAdd(&(params.quaternions_grad[offset_index].w),quaternions_grad.w);
            // atomicAdd(&(params.quaternions_grad[primitive_index].x),quaternions_grad.x);
            // atomicAdd(&(params.quaternions_grad[primitive_index].y),quaternions_grad.y);
            // atomicAdd(&(params.quaternions_grad[primitive_index].z),quaternions_grad.z);
            // atomicAdd(&(params.quaternions_grad[primitive_index].w),quaternions_grad.w);
    }
    dot_diff_col_ray_col_back=dot_diff_col_ray_col_back_after;
    transmittance_backward=transmittance_aft_buffer;
    }
    tbuffer+=slab_spacing;
    }
    }
}


extern "C" __global__ void __miss__ms_ah()
{

}

extern "C" __global__ void __anyhit__ah() {
    const unsigned int num_primitives = optixGetPayload_0();

    if (num_primitives >= params.max_prim_slice) {
        //printf("The number of primitives is greater than the maximum number of spheres per ray\n");
        optixTerminateRay();
        return;
    }

    const uint3 idx = optixGetLaunchIndex();
    const unsigned int idx_ray= idx.x + idx.y * params.image_width;
    const unsigned int current_gaussian_idx = optixGetPrimitiveIndex();

    params.hit_prim_idx[idx_ray * params.max_prim_slice + num_primitives] = current_gaussian_idx;


    optixSetPayload_0(num_primitives + 1);
    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__ms_ch()
{
    optixSetPayload_0(__float_as_int(-1.0f));
}


extern "C" __global__ void __closesthit__ch()
{
    float t=optixGetRayTmax();
    optixSetPayload_0(__float_as_int(t));
}