---
type: note
aliases: 
tags:
  - python
  - lightsails
---
## Ideas
Split **`ParticleSystem`** into **`ParticleSystem`**, **`ParticleSystemPlotter`** and **`ParticleSystemBuilder`**
## Plotting function
\[[Git Commit](https://github.com/Markk116/LightSailSim/commit/b47356530f128b9bac8f723357c5837054208223)]

Using linecollection 3d to speed up plotting
- ![[WhatsApp Image 2023-11-02 at 15.39.18_321bdfde.jpg]]
- ![[Pasted image 20231102154512.png|300]]
### Further improvements
- [ ] Refine plotting function to give users more control of how it looks
- [ ] Refine plotting function to be in line with the [recommended signature](https://stackoverflow.com/questions/43925337/matplotlib-returning-a-plot-object)
- [ ] Refine plotting function to color the nodes by their strain
	- Stress is harder, because I'd need to define an area to distribute the force over
- [ ] For plotting large meshes this is also slow. Consider plotting triangulation from point cloud instead. 


## Finding surface
\[[Git commit](https://github.com/Markk116/LightSailSim/commit/644118c3c4b1bc49142af4ba37ea5e74efbeb14d)]
### Notes on proof of concept
I think I should take strain into account
	- Options I have so far:
		- Feed nodes to scipy interpolater, interpolate surface, probe surface at each node point
			- Hard to define the area of each section then
		- Alexanders method (requires square mesh): define element as bounded by four nodes, calculate orientation and area of element, distribute forces evenly over surrounding nodes. 
		- Somehow generate the voronoi diagram for the point cloud
			- From [merigotVoronoiBasedCurvatureFeature2011](zotero://select/library/items/TVTHR9E5)
			- “the most common method of estimating principal curvatures and principal curvature directions on point clouds is to locally t a polynomial of a given degree, and to analytically compute principal curvatures of this tted polynomial (Mérigot et al., 2011, p. 15)
			- The complexity of this is too high for me right now
	- So having done some research. The problem class is called [Triangulation](https://en.wikipedia.org/wiki/Triangulation_(geometry))
		- A good option for doing this is using [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation)
			- Which is implemented in Scipy: 
				- [`scipy.spatial.Delaunay`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html)
				- it uses an external library to accomplish this, Qhull. It's important I pass the right options
				- Most notably, I think Qhull drops coplanar points, that way of course you represent the point cloud with as few triangles as possible = efficient
				- However, I need those triangles for force calculation
				- so I need to pas "QJ"
			- I  can then distribute the forces of each triangle evenly over its points 
				- I'm afraid this scales kinda poorly
		- Another option is constructing the voronoi diagram using the scipy method for that
			- However for 3d I will get a set of volumes. I would then have to slice that into the right sections. 
			- Probably better just to stick with triangulation. 
- Made a testfile, then was like, wonder what it looks like, let's plot it
	- [Found matplotlib tutourial](https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html)
	- Turns out matplotlib is also calling Delaunay when plotting point clouds 
	- ![[Pasted image 20231103113153.png]]
	- Plotting takes about the same amount of time as new method, but is much more responsive to mouse inputs. 
- Turns out I misread, [`scipy.spatial.Delaunay`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html) turns a 3d point cloud into tetrahedra, not triangles. 
	- GPT-4 recommends using [`scipy.spatial.ConvexHull`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) instead
	- But gives me just the bounding box, not all the triangles.
	- ![[Pasted image 20231103124011.png]]
	- Next option is to project the mesh onto a 2D plane, take the triangles there and reflate into 3D
	- that works
	  ![[Pasted image 20231103123904.png]]
	- But is obviously very fragile, as it requires the mesh to be projectable onto a plane. The way Alexander does it is more robust. 
	- I think I will move forward with this because it is fast, but I will have to keep in mind that this limitation exists, and document it. 
- Okay, now to calculate the areas and orientations
	- Cross product of two vectors gives me the area of the parallelogram
	- Divide by 2 for triangle area
	- and cross product also provides unit vector if I normalise it. 
	- Got it!
	- ![[Pasted image 20231103133313.png]]![[Pasted image 20231103133321.png]]
	- Again, in terms of robustness it totally sucks:
	- ![[Pasted image 20231103133551.png]]
	- What's happening here is that it's shrinking in y due to the Poisson effect, and so it's filling in those little side triangles. 
	- So this means I should initialise the triangles before simulating even 1 step. and then just update the positions after that
	- Which means I need a `__initialize_triangles()` that get's called only first time. 
- Now to divide the areas and orientations over each connected node!
	- Not sure if we can do this with numpy magic, I think we have to itterate which will be slow
	- Let's think about it. Goals:
		- Either
			- return area vector
			- or
			- return unit vector and area list
		- Former way more elegant
	- so we have a list of area vectors for each simplice
	- we could build a huge matrix that does the operation
	- iterate once, eat the suck and then be relatively fast after that
	- we can iterate over the nodes
		- if the node is part of a simplice
			- set the elements at the right indices to 1/3
	- To apply use np.matmul
	- and then reshape input to be nx3 array
	- This would be much more elegant with tensors, but I'm afraid my brain too smooth for that
		- Plus it'd be hell to maintain for the next person
	- Okay flattening, using np.matmul and then reshaping kinda sorta worked ![[Pasted image 20231103153305.png]]
	- Basically, the points with more vertexes get a disproportionate amount of the area ![[Pasted image 20231103153635.png]]
	-  I think we can fix this based on the angle of the triangle at the node
		- the 90 deg angle should get half the triangle, the 45 degree angles should get a quarter each.
		- for triangles that are all 60 degrees we get a third for each.
		- so it's just angle/180. 
		- I realise this is true for planar geometry, but is different in 3d space
		- but the triangle is still planar and that's really the unit of analysis here
	- For the node that we randomly picked to calculate the area from we have the vectors so we can use the inner product rule $\cos(\theta) = \frac{x \cdot y}{|x||y|}$
		- we can then use law of sines to find an other angle $\frac{\sin\alpha}{a}=\frac{\sin\beta}{b}$
	- A little bit of messing around later I have: ![[Pasted image 20231103163443.png]]
		- Victory!
		- Ooh ooh, **I could use the length of each of these vectors to set the particle mass accurately**.
- Next is implementing it in **`ParticleSystem`**
	- I will take the code I have and split it into multiple functions
	- I need to make a new private attribute to hold the conversion matrix
	- I need to make a `__instantiate_surface()` method that builds the mesh when the system is initialised 
		- This needs optional arguments to handle setting the mass
		- Like an optional surface density argument function
		- could check for lambda function with type(rho)== types.FunctionType
	- I need a `calulate_surfaces()` method

high level description so far:
- Project mesh on z plane
- triangulate using Delaunay's algorithm
- calculate initial areas and angles of triangles
- build conversion matrix for distributing areas over nodes

### Notes on implementation into **`ParticleSystem:`**
- Split into two functions
	- First is `__initialize_find_surface()` which performs triangulation and sets up conversion matrix for surface calculation
	- Second is `find_surface()`, which checks if the init has been performed, does so if that is not the case, and calculates the areas
- Also added a new plotting function: `plot_triangulated_surface()`. 
	- It plots triangulated surface for user inspection
	- ![[Pasted image 20231106121047.png|300]]![[Pasted image 20231106121452.png|300]]
	- Seems pretty fast too![[Pasted image 20231106121607.png]]

### Limitations
- ==!!! MESH MUST BE PROJECTABLE ONTO X-Y PLANE AT INIT!!!== 
	- For convenience could split more complicated systems into parts maybe. Like top and bottom half of an airfoil.
- ===Currently not taking changing element shape into account when distributing areas over the nodes===
	- but could rebuild the conversion matrix every n iterations.
- Assumes mesh with no holes in it. 

### Further improvements
- [ ] Find a more robust triangulation algorithm 
	- PyMesh is an option [PyMesh - Mesh Generation - Triangulation](https://pymesh.readthedocs.io/en/latest/api_mesh_generation.html?highlight=triangulate)
		- But it features the same issues, even worse they only accept coplanar point clouds. At least they way I'm doing it now I am already handling the projection. 
		- Could cut the mesh into bits in a smart way and project parts on a dedicated plane, then reinflate them and assemble the whole.
		- Sounds like _hell_
- [ ] Allow projection onto arbitrary plane:
	- https://en.wikipedia.org/wiki/Vector_projection ![[Pasted image 20231106105614.png]]
- [ ] Add option to periodically rebuild conversion matrix to account of element shape distortion 
- [ ] Add support for holes
	- Holes must be given in nested list of points that make up the holes
	- Can then iterate over simplices and drop all those that are made up of just those points. 
- [ ] Update plotting function to make sure mesh is initialized 


## LaserBeam
\[[Git commit](https://github.com/Markk116/LightSailSim/commit/c17848a5232f9eaecee6a02e8ed3f8528c43d2e5)]
This one is pretty simple, just packaging lambda functions into a class. Representing polarization with a Jones vector. 
![[Pasted image 20231107142641.png]]

## **`OpticalForceCalculator`**
\[[Git commit](https://github.com/Markk116/LightSailSim/commit/8b4f9df3ee4495be5b506ccb341ea58be32a4acc)]
### Notes
#### Context
- This is a real big one
	- I've kinda been procrastinating this one a bit because it requires me making _decisions_ that have _long term consequences_
- Goal is to have a class that pulls together all the optical information such that it can be called each timestep to provide the optical forces
- It interacts with the **`ParticleSystem`** because it shares the list of `Particle`s
- Why a class rather than a method of **`ParticleSystem`**?
	- PS is growing a lot, starting to get really unwieldy
	- The optical model requires a bunch of setup which would make setting up the PS even more annoying
	- Better to sequence it into smaller steps.
	- Helps with the flow for the users and facilitates use of Builder Design Pattern later.

#### The math
- So what is it actually doing?
	- Goal is to take incident ray, find the reflection and from that calculate the resulting force
	- the incident rays we find by polling the LaserBeam.intensity_profile(x,y) at all node locations
		- We assume it's always going to positive z
	- Then we have to account for all the possible cases of optical materials in the particles
		- Maybe we extract these on OpticalForceCalculator.\_\_init__() 
Running through the general math:
$$p_{photon}= \frac{h}{\lambda}$$
$$F_{optical} = \frac{\Delta p }{\Delta t}$$
$$\frac{\Delta n_{photons}}{\Delta t} = \frac{P}{E_{photon}}=\frac{P\lambda}{hc}$$
$$\Delta p = \Delta n_{photons}* p_{photon}$$
$$F_{optical} = \frac{\Delta n_{photons}* p_{photon}}{\Delta t}=\frac{P\lambda}{hc}\frac{h}{\lambda}= \frac{P}{c}$$
Lol so we just come out to Power by speed of light. That's convenient. 
$$\frac{\Delta F_{optical}}{\Delta Area}= \frac{P}{c\Delta Area}= \frac{I}{c}$$
So for each element $$F_i = \frac{A_i I_i}{c} \text{, where } I_i = I(x_i,y_i)[0,0,1], A_i = a_i [n_{x,i},n_{y,i},n_{z,i}] $$
#### Cases

##### Specular reflection
- First Specular reflection: 
	- we know force is normal to surface area
	- the force is a function of the cosine of the angle between the normal vector and the incident light ray
	- we can find this by using the dot product rule $A\cdot B = |A||B| \cos{\theta}$
-  Got the right direction, now the magnitude![[Pasted image 20231122115701.png]]
	- I think part of the problem is the way I define the laser beam intensity. 
		- Because it is kinda sloppy and so I don't have a nice reference
		- I need to make a small library of testcases that I can borrow things from. Like standard instantiated meshes. 
		- Squares, circles, domes
- For now, I have set the beam to 100 GW /  (10m * 10m)  = 10 GW/m2
	- So the maximal total force is 100 GW/c = 333.6 N
	- Got a good distribution for a flat sheet:![[Pasted image 20231122120230.png]]
	- Actually, good enough for guvmn't work!![[Pasted image 20231122120608.png]]
- Okay, happy with that, for validation I gotta run this with angled planes and domes 
- Doesn't run too slow either, DEFO could use some optimization but it won't break the bank right now. if really bad I could just update it every nth iteration anyway. Maybe update only when the kin damping is enacted, I have a feeling like there is something special case like about that.  ![[Pasted image 20231122121159.png]]
##### Axicon mirror
- Next case: 
	- One issue I have is that in a particle system I have poor reference for 'orientation'
	- The axicon is defined by having a directing angle, but this is not the only parameter of importance, because the 'azimuth' of the direction angle also matters. 
	- If I define this w.r.t. the absolute coordinates, it would break on rotation of the whole.
	- So maybe I have to define a local coordinate system as well. 
		- Can I use the triangulation I made already for this? 
		- maybe it's best for now to eschew this now and accept it as a limitation. 
		- This will only cause errors with large in plane translations, but I expect the dominating factor to be out of plane translations and rotations. 
	- So keeping it simple: Will effectively rotate the normal vector of the area 
		- ~~~First rotate from z+~~~ 
		- Let's keep it actually simple and just do x,y rotations 
			- Could just attach a rotation object to each particle
		- Okay got it to work, what actually took the longest was getting the damn testcase to work because I kept getting twisted up with the rotations![[Pasted image 20231201133623.png]]
		- But yea, got a multi pattern sail now
####  On splitting cases:
- How to split cases
	- I am a bit at a loss for how to do this efficiently
	- Like, I don't want to iterate over the particle list and calculate the forces individually because that will eat performance 
	- So I want to do it using np.arrays
	- my first idea was using a[a\==x] notation to make a mask for each type and then calculating them separately, but I don't know how to put them together after doing the calculations. 
		- Unless I add a column with indices first, like pandas has built in
		- then I can split it, work on it and put it together, then sort, finally cut off the index column
		- Why not just use pandas then? Don't want to deal with extra package if I only need it for one thing. 
- Had a chat with GPT-4, realised during that passing references is probably a better idea than passing copies of sub arrays, also for memory performance
	- Like so ![[Pasted image 20231130120904.png]]
- So I'm going to loop over the enum, then create different views and make a dict of masks
	- Currently doing this again every time, should put this in an `__init_` method
### Limitations
==Breaks when sail is allowed to rotate around z==

### Further improvements
- [ ] Make init function for splitting cases to speed up program
- [ ] Split stability calculations away from force calculations 
- [ ] Add rotation dependence for axicon mirror PhC
- [ ] Check if rotation is applied after cosine factor calculation rather than before

### V&V notes
- Run following cases:
	- Flat square sheet
	- Flat circle
	- Flat sheet at 45 degree angles to the laserbeam


## Stability Coefficients
- Adding it to the optical force calculator because it seems to fit there nicely. 
- Need to displace and rotate the particle system (reversibly) 
- Procedure (pseudocode):
	- Find center of mass
	- start loop for each displacement
		- displace sail
		- calculate raw optical forces
		- find net displacing and tilting reaction
		- find stability coefficients via $k_q = \frac{\Delta F_q}{\Delta q}$
		- un-displace sail
	- end loop
	- return values
- Implementation wise
- I think it might actually be nice to split it in a few functions:
	- one that displaces the particles
	- one that un-displaces them
	- one that calculates COG
	- one that puts it all together
- I am now thinking about the 2d example, but I have 2 translations and 2 rotations to worry about...
	- x, y displacement
	- rx, ry rotations
- Got the 2D case for displacement to work
	- ![[Pasted image 20231124121043.png]] ![[Pasted image 20231124121057.png]]
- And now in 3D and actually around the center of mass![[Pasted image 20231124123431.png]]
- Now let's calculate the restoring forces. 
	- For translation we can simply sum over the forces
	- For rotation we need to do some cross products.
- NOTE: Currently I added this all into the **`OpticalForceCalculator`**. It's growing a lot and so I have regrets about that. Here's my issue: 
	- I don't want to make it 'general mesh' tools because it has dependencies on the optical force things
	- I can split it into a 'stability' class, but I feel there are similar issues there
	- Maybe you'd wrap the optical forces in a generic external forces class that collects all external forces (if there are any other), which is then required on init of the stability tools. 
- Okay got the basic logic running, seems to check out on the gut feel level![[Pasted image 20231124145008.png]]



### V&V notes
- Immediately some things I have to do come up:
	- Write secondary function to test if it's linear
		- What criteria? curvature?
		- There's linear with respect to displacement, but should also check superposition holds. ( $Disp(x,y) = Disp(x)+Disp(y)$ )
	- displace only orthogonally or do random direction sampling?
- There is another big issue. 
	- Let's say we displace a small amount 
	- This will produce forces on the sail
	- This will deform the sail
	- Do I want the stability coefficients at the instant of displacement or should I let it settle in the new configuration first?
		- I guess it depends on the settling time, if the settling takes very long it's arguable that the sail will have moved again so it's not relevant.
	- Which means that ==this only works for very stable sails== 
	- Should be investigated #later

## Optimization
This is a plot of the A matrix from the simulate function:
- It's real sparse
  ![[Pasted image 20231106150757.png|450]]
- Might offer speed up to use [sparse matrix formats](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- Small test yields that this helps once a matrix becomes bigger than 250x250. Already really significantly so for a 1000x1000 matrix.
	- [BICGSTAB](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicgstab.html) takes a sparse matrix
	- And that speeds it up a lot actually (upper matrix multiplication, lower BicGStab)
	- ![[Pasted image 20231106160011.png]]
	- If we also build the system matrix more cleverly using the [sp.sparse tools](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.html#scipy.sparse.dok_array) we can probably get better times even. 

## Boundary conditions and reaction forces
\[[commit: Plane](https://github.com/Markk116/LightSailSim/commit/cfeae1d8009c984f8a7810ab8fba9180254458de)]
\[[commit: Line](https://github.com/Markk116/LightSailSim/commit/4829e3a195e5dc29fd1391752744c5d6589cc683)]
Putting these two together because I will have they are interrelated
Before we start cutting the patient open, let's MRI his heart.
- the `__fixed` property is set on Particle at init, it is currently a `bool` 
	- It is then used in `update_pos` and `update_vel`
- It is also used in `PS.simulate` where it is used to zero out parts of the A matrix
	- ![[Pasted image 20231201134821.png]]
	- also when calculating external forces
	- and setting colour when plotting

Now how to approach this:
- My first idea is to replace the bool with a length 3 vector
	- If it's \[1, 1, 1] then it should do nothing
		- Maybe I can set a bool for that
	- If it's \[0, 0, 0] then it should fix the particle completely 
	- Then I can mix and match for fixing in different DOF's 
		- But here's where the magic comes in: if we put floats $\epsilon[0,1]$ we can fix things to planes. 
		- Effectively we would be projecting the motion onto the fixing plane and only allowing that
		- How to handle the projection?
		- How to alter the A matrix of jacobian to accurately reflect this

**Particle level**
Okay, second pass:
- Effectively I'm projecting the motion onto a plane described by a vector. So the above notation won't work super well for fixing and having free particles
- So I'll keep the `Particle.__fixed` and just add `Particle.__constraint` to hold the actual constraint, with a handy `Particle.constraint_projection_matrix` to pre-bake that
- Since I'm projecting I also have to project the updated velocities and positions for the particles (almost forgot this)
	- But I should only project the change in velocity
	- So I subtract the current and matmul with the matrix nm
- Okay, this part seems to check out ![[Pasted image 20231204143045.png]]
- waitwaitwait I've now got a plane and a fixed constraint but no line constraint!
	- Fixed that now

**System level**
- Now let's work on the System level problem. Let's first set up the matrix to multiply the whole system
	- Hmm, what do I mean by this exactly? 
		- I don't have to make one for the particles, because the update functions are taking care of that already. Would be faster to do it system-level but way less robust
		- So I guess I mean the matrix with which to multiply the jacobians
		- For example take a particle in 2D that is fixed to the plane normal to (1,1). Now if we pull this particle in the -x direction scaling the jacobian is not enough. Because the particle will comply to some degree, but also slide along the plane in the +y direction. So there is a coupling that arises. 
	- I also realise that this isn't strictly needed, the code will just run fine like it is now. But the whole point of the bigstab is giving it information from the future with in the jacobian, so I think it will just converge much better if I do it correctly. 
	- So, first understanding how the jacobians are assembled:
		- for each spring damper the jacobians are found
		- These are then assembled into two matrices, one for dx, other for dv
		- each J is used twice, once for the for between p1 and p2, and another time for p2 and p1
		- The question is, is the effect of the constraint symmetric? 
	- You know what, let's shelve this for now under 'further improvements'
- Okay, when working on the reaction forces I found out that the line and plane constraints don't actually allow motion. This is because the fix particles are blanked out in the creation of the A matrix of the simulate function. 
	- It should not zero them out
	- Two places I could intervene instead
		- don't zero things out but multiply appropriate things with projection
		- OR
		- don't zero things out and multiply jacobians appropriately. 
### Further Improvements
- [ ] integrate the constraints into the jacobians

## ParticleSystem.find_reaction_forces()
\[[commit](https://github.com/Markk116/LightSailSim/commit/0179f0ca5aefc52f14c1eb93a8a49462135fd54c)]
So, want to quickly retrieve reaction forces. 
Get forces, reshape into 3d array, then filter using fixed list.
Now we have to do something a little more in order to find the right forces for the case that we have less than 3DOF constrained. Using the projection matrix we can find the kinematically enabled forces, and then we can subtract that from the original to find what the reaction forces are. 
This works!



## Mesher
So, I got sick of doing this manually so I want to write something for this
### The architecture.
- I took one look at pymesh and decided it wouldn't work for me because I don't have much control over the actually mesh shape
- But the workflow of defining shapes and then meshing them makes sense
- so this is what I have in mind:
	- A main `Mesher` class to hold the mesh
		- `.add_shape` methods to add shapes to a list, together with the info about how it should be meshed 
		- `.mesh_shapes` to mesh them
	- A `Geometry` class as abstract base for
		- `Rectangle`
			- `Square` can be a handle for `Rectangle`
		- `Elipse`
			- `Circle` can be a handle for `Ellipse`
		- Can add `Polygon` later
	- Each `Geometry` child class will implement methods for meshing itself in different ways.
		- For example `Rectangle.mesh_square()`
	- 

Ideas:
- Ellipse can secretly call rectangle and perform a coordinate transformation on the final mesh to make it circular. 

What are the different parameters for meshing?
- So, let's take a rectangle. I can imagine splitting it up into smaller rectangles. 
	- Each can then contain a stretched unit cell of the mesh shape
- I'd need to know how many cells to split it into
	- So we need width, length and a scaling parameter for the mesh
	- How to define the scaling parameter? 
		- Preferably if we have two different rectangles the same inputs should yield the same result
		- maybe `goal_unit_cell_edge_length`?
		- 


How do we connect Multiple shapes?
- Don't think we need to so let's skip it for now. 