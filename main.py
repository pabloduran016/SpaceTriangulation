import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.collections as mp_coll
import matplotlib.text as mp_text
import matplotlib.axes as mp_axes
import matplotlib.figure as mp_figure
from mpl_toolkits.mplot3d import art3d, Axes3D
from typing import List, Optional as Op

FArray = npt.NDArray[np.float32]
IArray = npt.NDArray[np.int32]
Ax = mp_axes.Axes
Fig = mp_figure.Figure
Text = mp_text.Text

def calculate_normals(points: FArray, triangles: FArray) -> FArray:
    normals = np.zeros((triangles.shape[0], points.shape[1]), np.float32)
    for i in range(triangles.shape[0]):
        triangle = triangles[i, :]
        vs = np.zeros((triangles.shape[1], points.shape[1]))
        for j in range(vs.shape[0]):
            p0 = points[triangle[j]]
            pf = points[triangle[(j+1)%len(triangle)]]
            vs[j, :] = pf - p0
        assert np.all(np.sum(vs, axis=0) < 1e-10)
        normals[i] = n = 1/2 * np.cross(vs[0], vs[1])
        assert np.all([abs(np.dot(n, vs[i])) < 1e-10 for i in range(vs.shape[0])])
    return normals

def plot_segments(points: FArray, segments: IArray, ax: Axes3D) -> None:
    for i in range(segments.shape[0]):
        s = segments[i]
        ax.plot3D(
            points[s, 0], points[s, 1], points[s, 2], 
            linewidth=1,
            color="black",
            alpha=0.5,
        )

def plot_points(
    points: FArray, ax: Axes3D,
    *,
    selected_idx: Op[int],
    secondary_selection: List[int],
    plot_id: bool,
) -> mp_coll.PathCollection:
    colors = [
        "purple" 
        if i in secondary_selection else
        "blue"
        if i == selected_idx else 
        "red"
        for i in range(points.shape[0])
    ]

    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],  # pyright: ignore[reportArgumentType]
        marker=".",
        s=100,
        facecolor=colors,
        edgecolor="black",
        picker=5,
    )
    if plot_id:
        for i in range(points.shape[0]):
            x, y, z = points[i]
            if i == selected_idx:
                ax.text(x, y, z, f"p[{i}] = {x:.02f}, {y:.02f}, {z:.02f}")
            else:
                ax.text(x, y, z, f"p[{i}]")
    return scatter


def plot_triangles(
    points: FArray, triangles: IArray, normals: FArray, 
    ax: Axes3D,
    *,
    draw_segment_vectors: bool,
    draw_normals: bool,
    plot_id: bool,
) -> List[Text]:
    texts = []
    for i in range(triangles.shape[0]):
        triangle = triangles[i, :]
        if draw_segment_vectors:
            for j in range(triangle.shape[0]):
                p0 = points[triangle[j]]
                pf = points[triangle[(j+1)%len(triangle)]]
                v = pf - p0
                ax.quiver3D(
                    p0[0], p0[1], p0[2],
                    v[0], v[1], v[2],
                )
        center = np.mean(points[triangle], axis=0)
        if draw_normals:
            n = normals[i]
            ax.quiver3D(
                center[0], center[1], center[2],
                n[0], n[1], n[2],
                color="black",
                linewidth=0.5,
                arrow_length_ratio=0.1,
            )
        ax.add_collection(
            art3d.Poly3DCollection(
                [points[triangle]],
                facecolor="yellow",
                alpha=0.1,
            )
        )
        if plot_id:
            ax.text(center[0], center[1], center[2], f"t[{i}]")
    return texts


def calculate_vertices(points: FArray, tetrahedra: IArray) -> FArray:
    vertices = np.zeros((tetrahedra.shape[0], points.shape[1]), np.float32)  # List of (x, y, z) coordinates
    for i in range(tetrahedra.shape[0]):
        vertices[i] = np.mean(points[tetrahedra[i], :], axis=0)
    return vertices

def calculate_edges(triangles: IArray, tetrahedra: IArray) -> IArray:
    edges = []
    for i in range(tetrahedra.shape[0]):
        tet_i = tetrahedra[i, :]
        for j in range(tetrahedra.shape[0]):
            if j >= i:
                continue
            tet_j = tetrahedra[j, :]
            for k in range(triangles.shape[0]):
                if all(p in tet_i for p in triangles[k]) and all(p in tet_j for p in triangles[k]):
                    edges.append([i, j])
    return np.array(edges)

def plot_vertices(
    vertices: FArray, ax: Axes3D,
    *,
    plot_id: bool,
) -> None:
    for i in range(vertices.shape[0]):
        x, y, z = vertices[i]
        color = "green"
        ax.scatter(
            x, y, z,
            marker=".",
            s=100,
            facecolor=color,
            edgecolor="black",
            picker=5,
        )
        if plot_id:
            ax.text(x, y, z, f"v[{i}]")

def plot_edges(vertices: FArray, edges: IArray, ax: Axes3D) -> None:
    for i in range(edges.shape[0]):
        s = edges[i]
        ax.plot3D(
            vertices[s, 0], vertices[s, 1], vertices[s, 2], 
            linewidth=1,
            color="blue",
            alpha=0.5,
        )

def main() -> None:
    points = np.array([  # List of (x, y, z) coords
        [0, 0, 1],
        [1, 0, 0],
        [-1/2, np.sqrt(3)/2, 0],
        [-1/2, -np.sqrt(3)/2, 0],

        [-1.5, 0, 1],
    ])
    # points -= np.mean(points, axis=0)

    segments = np.array([  # List of (initial, final) indices in points
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],

        [0, 4],
        [2, 4],
        [3, 4],
    ])
    triangles = np.array([  # List of (p1, p2, p3) indices in points
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],

        [0, 2, 4],
        [0, 4, 3],
        [2, 3, 4],
    ])

    tetrahedra = np.array([  # List of (p1, p2, p3, p4) indices in points
        [0, 1, 2, 3],
        [0, 2, 3, 4],
    ])

    # vertices = calculate_vertices(points, tetrahedra)
    # edges = calculate_edges(points, triangles, tetrahedra)

    data = {
        "points": points,
        "segments": segments,
        "triangles": triangles,
        "tetrahedra": tetrahedra,
        "selected_idx": None,
        "secondary_selection": [],
        "scatter": [],
        "selection_mode": "none",
        "plot_id": False,
    }
    filename = "triangulation.npz"
    d = np.load(filename)
    for k in d.keys():
        data[k] = d[k]

    def update_plot():
        """Refreshes the scatter collection and updates colors based on selection."""
        # for s in data["scatters"]:
        #     s.remove()  # Clear old scatter object
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        # Highlight the selected point in red, others in blue
        vertices = calculate_vertices(data["points"], data["tetrahedra"])
        edges = calculate_edges(data["triangles"], data["tetrahedra"])
                
        plot_edges(vertices, edges, ax)
        plot_vertices(vertices, ax, plot_id=data["plot_id"])
        normals = calculate_normals(data["points"], data["triangles"])
        plot_triangles(
            data["points"], data["triangles"], normals, ax,
            draw_segment_vectors=False,
            draw_normals=False,
            plot_id=data["plot_id"],
        )
        plot_segments(data["points"], data["segments"], ax)
        data["scatter"] = plot_points(
            data["points"], ax,
            selected_idx=data["selected_idx"],
            secondary_selection=data["secondary_selection"],
            plot_id=data["plot_id"],
        )

        ax.text2D(0.05, 0.95, f"Selection Mode: {data['selection_mode']}", transform=ax.transAxes)
        fig.canvas.draw_idle()

    def on_pick(event):
        """Triggered when clicking on an existing point."""
        if event.artist == data["scatter"]:
            i = event.ind[0]
            if data["selection_mode"] == "segment" and data["selected_idx"] is not None and data["selected_idx"] != i:
                s = tuple(sorted([data["selected_idx"], i]))
                segments2 = []
                for j in range(data["segments"].shape[0]):
                    seg = tuple(data["segments"][j, :])
                    if np.all(seg == s):
                        s = None
                        continue
                    segments2.append(seg)
                if s is not None:
                    segments2.append(s)
                data["segments"] = np.array([[p0, p1] for p0, p1 in segments2])
            elif data["selection_mode"] == "triangle" and data["selected_idx"] is not None and data["selected_idx"] != i:
                if i in data["secondary_selection"]:
                    data["secondary_selection"].remove(i)
                if len(data["secondary_selection"]) < 2:
                    data["secondary_selection"].append(i)
                if len(data["secondary_selection"]) == 2:
                    triangle = [data["selected_idx"], *data["secondary_selection"]]
                    triangles2 = []
                    for j in range(data["triangles"].shape[0]):
                        tri = list(data["triangles"][j, :])
                        if all(p in triangle for p in tri):
                            triangle = None
                            continue
                        triangles2.append(tri)
                    if triangle is not None:
                        triangles2.append(triangle)
                    data["triangles"] = np.array([[p0, p1, p2] for p0, p1, p2 in triangles2])
                    data["secondary_selection"] = []
            elif data["selection_mode"] == "tetrahedra" and data["selected_idx"] is not None and data["selected_idx"] != i:
                if i in data["secondary_selection"]:
                    data["secondary_selection"].remove(i)
                if len(data["secondary_selection"]) < 3:
                    data["secondary_selection"].append(i)
                if len(data["secondary_selection"]) == 3:
                    tet = [data["selected_idx"], *data["secondary_selection"]]
                    tet2 = []
                    for j in range(data["tetrahedra"].shape[0]):
                        t = list(data["tetrahedra"][j, :])
                        if all(p in tet for p in t):
                            tet = None
                            continue
                        tet2.append(t)
                    if tet is not None:
                        tet2.append(tet)
                    data["tetrahedra"] = np.array([[p0, p1, p2, p3] for p0, p1, p2, p3 in tet2])
                    data["secondary_selection"] = []
            else:
                if i == data["selected_idx"]:
                    data["selected_idx"] = None
                else:
                    data["selected_idx"] = i
        update_plot()

    def on_click(event):
        """Triggered when clicking the background. Adds a new point at click depth projection."""
        # # Only act if it's a left click, inside the axes, and we didn't just pick a point
        # if event.button == 1 and event.inaxes == ax and ax.button_pressed == 1:
        #     # Check if we clicked empty space (ignores clicks meant for panning/rotating)
        #     if fig.canvas.widgetlock.locked(): 
        #         return
        #     
        #     # Approximate 3D position based on click projection midpoint
        #     if event.xdata is not None and event.ydata is not None:
        #         # Check if we are selecting an existing point first
        #         # If not, we add a point at the center of the Z-depth range
        #         z_mid = (ax.get_zlim()[0] + ax.get_zlim()[1]) / 2
        #         p = np.array([[event.xdata, event.ydata, z_mid]])
        #         data["points"] = np.concatenate([data["points"], p])
        #         data["selected_idx"] = data["points"].shape[0] - 1
        #         update_plot()

    def on_key(event):
        """Moves the selected point using Arrow Keys."""
        if event.key == 'A':  # Add point
            p = np.array([[0, 0, 0]])
            data["points"] = np.concatenate([data["points"], p])
            data["selected_idx"] = data["points"].shape[0] - 1
        elif event.key == 'esc':
            data["selection_mode"] = "none"
        elif event.key == 'S':
            data["selection_mode"] = "segment" if data["selection_mode"] != "segment" else "none"
            data["secondary_selection"] = []
        elif event.key == 'T':
            data["selection_mode"] = "triangle" if data["selection_mode"] != "triangle" else "none"
            data["secondary_selection"] = []
        elif event.key == 'V':
            data["selection_mode"] = "tetrahedra" if data["selection_mode"] != "tetrahedra" else "none"
            data["secondary_selection"] = []
        elif event.key == 'N':
            data["plot_id"] = not data["plot_id"]
        elif event.key == 'G':  # Toggle segment mode
            np.savez(
                filename,
                points=data["points"],
                segments=data["segments"],
                triangles=data["triangles"],
                tetrahedra=data["tetrahedra"],
            )
            print(f"Saved triangulation to file: {filename}")
        elif event.key == 'L':  # Toggle segment mode
            d = np.load(filename)
            for k in d.keys():
                data[k] = d[k]

        
        if data["selected_idx"] is not None:
            step = 0.1  # Movement speed increments
            dx = dy = dz = 0
            if event.key == 'left':
                dx -= step
            elif event.key == 'right':
                dx += step
            elif event.key == 'up':
                dy += step
            elif event.key == 'down':
                dy -= step
            elif event.key == 'shift+up':
                dz += step
            elif event.key == 'shift+down':
                dz -= step
            data["points"][data["selected_idx"], :] += [dx, dy, dz]

        update_plot()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    update_plot()
    # Connect the interactive event managers
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


if __name__ == "__main__":
    main()
