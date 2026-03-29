import numpy as np
import matplotlib.pyplot as plt

def debug_cbf(a, b, c, d, x_point, y_point):
    """
    The exact logic from your ACADOS / Renderer setup.
    f = a*w1^2 + b*w2^2 + c*w1 + d*w2 - 1
    h = -f
    """
    w1, w2 = x_point, y_point
    
    # The 'f' value from your visualization logic (F=-1)
    f_val = a*(w1**2) + b*(w2**2) + c*w1 + d*w2 - 1.0
    K = 1.0 + c**2 / (4 * a) + (d**2 / (4 * b))
    
    # The CBF value we are passing to Acados
    h = -f_val / K
    
    # Theoretical Center for reference: -c/(2a), -d/(2b)
    center_w1 = -c / (2 * a)
    center_w2 = -d / (2 * b)
    
    return h, (center_w1, center_w2)

def run_debugger(a=2.0, b=1.0, c=0.5, d=-0.2):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # Create a grid to show the actual ellipse boundary
    w1_range = np.linspace(-2, 2, 400)
    w2_range = np.linspace(-2, 2, 400)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    F = a*W1**2 + b*W2**2 + c*W1 + d*W2 - 1.0
    
    # Plot the boundary (where f = 0)
    ax.contour(W1, W2, F, levels=[0], colors='blue', linewidths=2)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    
    h_init, center = debug_cbf(a, b, c, d, 0, 0)
    ax.plot(center[0], center[1], 'bx', label='Calculated Center')
    
    ax.set_title(f"CBF Debugger\nParams: a={a}, b={b}, c={c}, d={d}")
    ax.set_xlabel("Local Frame w1 (e1)")
    ax.set_ylabel("Local Frame w2 (e2)")

    text_disp = ax.text(-1.9, 1.7, "Click anywhere to test CBF", 
                        bbox=dict(facecolor='white', alpha=0.8))

    def onclick(event):
        if event.inaxes != ax: return
        
        h, _ = debug_cbf(a, b, c, d, event.xdata, event.ydata)
        
        color = 'green' if h >= 0 else 'red'
        status = "SAFE (Inside)" if h >= 0 else "UNSAFE (Outside)"
        
        ax.plot(event.xdata, event.ydata, 'o', color=color)
        text_disp.set_text(f"Pos: ({event.xdata:.2f}, {event.ydata:.2f})\nCBF (h): {h:.4f}\nStatus: {status}")
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print(f"Debugger active. Center is at {center}")
    plt.show()

if __name__ == "__main__":
    # Test with your current coefficients
    # Example: narrow tube (high a, b) with an offset
    run_debugger(a=8.2, b=1.3, c=-7.2, d=-0.13)
