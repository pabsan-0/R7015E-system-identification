%%  REAL SYSTEM IMPLEMENTATION

g = 9.8;     b_f = 0;      m_b = 0.381;      l_b = 0.112;   I_b = 0.00616;
m_w = 0.036; l_w = 0.021;  I_w = 0.00000746; R_m = 4.4;     L_m = 0;
b_m = 0;     K_e = 0.444;  K_t = 0.470;

gamma_11 = ( (I_w)/(l_w) + l_w * m_b + l_w * m_w );
gamma_12 = + m_b * l_b * l_w;
alpha_12 = - ( (K_e * K_t)/(R_m) + b_f ) / (l_w);
alpha_14 = + ( (K_e * K_t)/(R_m) + b_f );
gamma_21 = m_b * l_b;
gamma_22 = ( I_b + m_b * l_b^2 );
alpha_22 = ( (K_e * K_t)/(R_m) + b_f ) / (l_w);
alpha_23 = m_b * l_b * g;
alpha_24 = - ( (K_e * K_t)/(R_m) + b_f );
beta_11 = + (K_t)/(R_m); beta_12 = + l_w;
beta_21 = - (K_t)/(R_m); beta_22 = + l_b;
delta = gamma_11 * gamma_22 - gamma_12 * gamma_21;
a_22 = ( gamma_22 * alpha_12 - gamma_12 * alpha_22) / delta;
a_23 = ( - gamma_12 * alpha_23) / delta;
a_24 = ( gamma_22 * alpha_14 - gamma_12 * alpha_24) / delta;
a_42 = (- gamma_21 * alpha_12 + gamma_11 * alpha_22) / delta;
a_43 = ( + gamma_11 * alpha_23) / delta;
a_44 = (- gamma_21 * alpha_14 + gamma_11 * alpha_24) / delta;
b_21 = ( gamma_22 * beta_11 - gamma_12 * beta_21) / delta;
b_22 = ( gamma_22 * beta_12 - gamma_12 * beta_22) / delta;
b_41 = (- gamma_21 * beta_11 + gamma_11 * beta_21) / delta;
b_42 = (- gamma_21 * beta_12 + gamma_11 * beta_22) / delta;

A = [0 1 0 0; 0 a_22 a_23 a_24; 0 0 0 1; 0 a_42 a_43 a_44];
B = [0; b_21; 0; b_41];
C = [0 0 1 0];
D = zeros(1,1);

% cleaning useless variables
clear gamma_11; clear gamma_12; clear alpha_12; clear alpha_14;
clear beta_11;  clear beta_12;  clear gamma_21; clear gamma_22;
clear alpha_22; clear alpha_23; clear alpha_24; clear beta_21;
clear beta_22;  clear delta;    clear a_22;     clear a_23; 
clear a_24;     clear b_21;     clear b_22;     clear a_42;     
clear a_43;     clear a_44;     clear b_41;     clear b_42;

fprintf('Loaded real system parameters.\n');