#define WIDTH 1024
#define HEIGHT 1024

#define DX	0.03125f // (1 / 32)
#define RDX 32.0f      // (1 / DX)
#define HALFDX 0.15625f // (0.5f * DX)
#define VISCOSITY 1.0f

__kernel void UpdateVelocityBoundaries(__global float2* m_VelocityInput) {

    int idx = get_global_id( 0 );

	const float scale = -1.0f;
    if ( idx < WIDTH) {
        
		m_VelocityInput[idx + 0 * WIDTH] = m_VelocityInput[idx + 1 * WIDTH] * scale;
		m_VelocityInput[idx + (HEIGHT - 1) * WIDTH] = m_VelocityInput[idx + (HEIGHT - 2) * WIDTH] * scale;
    }

    if (idx < HEIGHT) {
		// Update the boundaries.
		m_VelocityInput[0 + idx * WIDTH] = m_VelocityInput[1 + idx * WIDTH] * scale;
		m_VelocityInput[(WIDTH - 1) + idx * WIDTH] = m_VelocityInput[(WIDTH - 2) + idx * WIDTH] * scale;
    }
};

__kernel void AdvectVelocity(float dt, __global float2* m_VelocityInput, __global float2* m_VelocityOutput) {

    int x = get_global_id( 0 );
    int y = get_global_id( 1 );

    const float fWidth = (float)WIDTH;
    const float fHeight = (float)HEIGHT;

    float2 pos = (float2)(x, y) - dt * RDX * m_VelocityInput[x + y * WIDTH];

    int stx = (int)clamp(floor(pos.x), 0.0f, fWidth - 1.0f);
    int sty = (int)clamp(floor(pos.y), 0.0f, fHeight - 1.0f);
    int stz = (int)clamp(stx + 1.0f, 0.0f, fWidth - 1.0f);
    int stw = (int)clamp(sty + 1.0f, 0.0f, fHeight - 1.0f);

    float2 t = (float2)(clamp(pos.x - stx, 0.0f, 1.0f), clamp(pos.y - sty, 0.0f, 1.0f));

    float2 v1 = m_VelocityInput[stx + sty * WIDTH];
    float2 v2 = m_VelocityInput[stz + sty * WIDTH];
    float2 v3 = m_VelocityInput[stx + stw * WIDTH];
    float2 v4 = m_VelocityInput[stz + stw * WIDTH];

    m_VelocityOutput[x + y * WIDTH] = mix(mix(v1, v2, t.x), mix(v3, v4, t.x), t.y);
};

__kernel void DiffuseVelocities(float dt, __global float2* m_VelocityInput, __global float2* m_VelocityOutput) {

    float alpha = (DX * DX) / (VISCOSITY * dt);
    float rBeta = 1.0f / (alpha + 4.0f);

    int x = get_global_id( 0 );
    int y = get_global_id( 1 );
	
    int stx = clamp(x - 1, 0, WIDTH - 1);
    int sty = clamp(y - 1, 0, HEIGHT - 1);
    int stz = clamp(x + 1, 0, WIDTH - 1);
    int stw = clamp(y + 1, 0, HEIGHT - 1);

    // Retrieve the four samples.
    float2 xL = m_VelocityInput[stx + y * WIDTH];
    float2 xR = m_VelocityInput[stz + y * WIDTH];
    float2 xB = m_VelocityInput[x + sty * WIDTH];
    float2 xT = m_VelocityInput[x + stw * WIDTH];

    // Sample b from the center.
    float2 bC = m_VelocityInput[x + y * WIDTH];

    // Evaluate the Jacobi iteration. 
    m_VelocityOutput[x + y * WIDTH] = (xL + xR + xB + xT + alpha * bC) * rBeta;
};

__kernel void ComputeDivergence(__global float2* m_VelocityInput, __global float* m_DivergenceInput) {
    int x = get_global_id( 0 );
    int y = get_global_id( 1 );

    int stx = clamp(x - 1, 0, WIDTH - 1);
    int sty = clamp(y - 1, 0, HEIGHT - 1);
    int stz = clamp(x + 1, 0, WIDTH - 1);
    int stw = clamp(y + 1, 0, HEIGHT - 1);

    float2 wL = m_VelocityInput[stx + y * WIDTH];
    float2 wR = m_VelocityInput[stz + y * WIDTH];
    float2 wB = m_VelocityInput[x + sty * WIDTH];
    float2 wT = m_VelocityInput[x + stw * WIDTH];

    m_DivergenceInput[x + y * WIDTH] = HALFDX * ((wR.x - wL.x) + (wT.y - wB.y));

}

__kernel void ComputePressure(__global float* m_PressureInput, __global float* m_PressureOutput, __global float* m_DivergenceInput) {
    int x = get_global_id( 0 );
    int y = get_global_id( 1 );

    float alpha = -1.0f * (DX * DX);
	float rBeta = 0.25f;

    int stx = clamp(x - 1, 0, WIDTH - 1);
    int sty = clamp(y - 1, 0, HEIGHT - 1);
    int stz = clamp(x + 1, 0, WIDTH - 1);
    int stw = clamp(y + 1, 0, HEIGHT - 1);

    // Retrieve the four samples.
    float xL = m_PressureInput[stx + y * WIDTH];
    float xR = m_PressureInput[stz + y * WIDTH];
    float xB = m_PressureInput[x + sty * WIDTH];
    float xT = m_PressureInput[x + stw * WIDTH];

    // Sample b from the center.
    float bC = m_DivergenceInput[x + y * WIDTH];

    // Evaluate the Jacobi iteration. 
    m_PressureOutput[x + y * WIDTH] = (xL + xR + xB + xT + alpha * bC) * rBeta;
}

__kernel void UpdatePressureBoundaries(__global float* m_PressureInput) {
    int idx = get_global_id( 0 );

	const float scale = -1.0f;
    if ( idx < WIDTH) {
        
		m_PressureInput[idx + 0 * WIDTH] = m_PressureInput[idx + 1 * WIDTH] * scale;
		m_PressureInput[idx + (HEIGHT - 1) * WIDTH] = m_PressureInput[idx + (HEIGHT - 2) * WIDTH] * scale;
    }

    if (idx < HEIGHT) {
		// Update the boundaries.
		m_PressureInput[0 + idx * WIDTH] = m_PressureInput[1 + idx * WIDTH] * scale;
		m_PressureInput[(WIDTH - 1) + idx * WIDTH] = m_PressureInput[(WIDTH - 2) + idx * WIDTH] * scale;
    }
}

__kernel void SubtractPressureGradient(__global float* m_PressureInput, __global float2* m_VelocityInput) {
    int x = get_global_id( 0 );
    int y = get_global_id( 1 );

    int stx = clamp(x - 1, 0, WIDTH - 1);
    int sty = clamp(y - 1, 0, HEIGHT - 1);
    int stz = clamp(x + 1, 0, WIDTH - 1);
    int stw = clamp(y + 1, 0, HEIGHT - 1);

    float pL = m_PressureInput[stx + y * WIDTH];
    float pR = m_PressureInput[stz + y * WIDTH];
    float pB = m_PressureInput[x + sty * WIDTH];
    float pT = m_PressureInput[x + stw * WIDTH];

    m_VelocityInput[x + y * WIDTH] = m_VelocityInput[x + y * WIDTH] - HALFDX * (float2)(pR - pL, pT - pB);
}

__kernel void UpdateColorBoundaries(__global float4* m_ColorInput) {
    int idx = get_global_id( 0 );

	const float scale = 0.0f;
    if ( idx < WIDTH) {
        
		m_ColorInput[idx + 0 * WIDTH] = m_ColorInput[idx + 1 * WIDTH] * scale;
		m_ColorInput[idx + (HEIGHT - 1) * WIDTH] = m_ColorInput[idx + (HEIGHT - 2) * WIDTH] * scale;
    }

    if (idx < HEIGHT) {
		// Update the boundaries.
		m_ColorInput[0 + idx * WIDTH] = m_ColorInput[1 + idx * WIDTH] * scale;
		m_ColorInput[(WIDTH - 1) + idx * WIDTH] = m_ColorInput[(WIDTH - 2) + idx * WIDTH] * scale;
    }
}

__kernel void AdvectColors(float dt, __global float4* m_ColorInput, __global float4* m_ColorOutput, __global float2* m_VelocityInput) {
    int x = get_global_id( 0 );
    int y = get_global_id( 1 );

    float fWidth = (float)WIDTH;
    float fHeight = (float)HEIGHT;

    float2 pos = (float2)(x, y) - dt * RDX * m_VelocityInput[x + y * WIDTH];

    int stx = (int)clamp(floor(pos.x), 0.0f, fWidth - 1.0f);
    int sty = (int)clamp(floor(pos.y), 0.0f, fHeight - 1.0f);
    int stz = (int)clamp(stx + 1.0f, 0.0f, fWidth - 1.0f);
    int stw = (int)clamp(sty + 1.0f, 0.0f, fHeight - 1.0f);

    float2 t = (float2)(clamp(pos.x - stx, 0.0f, 1.0f), clamp(pos.y - sty, 0.0f, 1.0f));

    float4 v1 = m_ColorInput[stx + sty * WIDTH];
    float4 v2 = m_ColorInput[stz + sty * WIDTH];
    float4 v3 = m_ColorInput[stx + stw * WIDTH];
    float4 v4 = m_ColorInput[stz + stw * WIDTH];

    m_ColorOutput[x + y * WIDTH] = mix(mix(v1, v2, t.x), mix(v3, v4, t.x), t.y);
}