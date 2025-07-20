No reasoning model prompt:

```python
f'''<objective>
Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model. Generate only the JSON file, no other text.
</objective>

<instruction>
You will be given a natural language description of a CAD design task. Your goal is to convert it into a structured JSON representation, which includes sketch geometry and extrusion operations.
The extrusion <operation> must be one of the following:
1. <NewBodyFeatureOperation>: Creates a new solid body.
2. <JoinFeatureOperation>: Fuses the shape with an existing body.
3. <CutFeatureOperation>: Subtracts the shape from an existing body.
4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.
Ensure all coordinates, geometry, and extrusion depths are extracted accurately from the input.
</instruction>

<description>
{prompt}
</description>'''
```

Reasoning model prompt:

```python
f'''<objective>
Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model based on the description provided. The output should include a reasoning section within the <think> tag and the corresponding JSON in the <json> tag. Do not provide any additional text outside of the tags.
</objective>

<instruction>
You will be given a natural language description of a CAD design task enclosed within <description> </description>. Your task is to:
1. Analyze the description and extract the relevant geometric and extrusion information.
2. In the <think> tag, explain how you derived each field and value in the JSON from the description. This includes the geometric properties (e.g., coordinates, shapes) and extrusion operations. The reasoning should clarify how the geometry is mapped to the JSON structure and the chosen extrusion operation.
3. Based on the reasoning in the <think> tag, generate the corresponding JSON structure for the CAD model in the <json> tag.

The extrusion <operation> must be one of the following:
1. <NewBodyFeatureOperation>: Creates a new solid body.
2. <JoinFeatureOperation>: Fuses the shape with an existing body.
3. <CutFeatureOperation>: Subtracts the shape from an existing body.
4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.
</instruction>

<description>
{prompt}
</description>'''
```