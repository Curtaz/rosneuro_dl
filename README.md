# rosneuro_dl
Toolbox for building nn-based decoders, integrated with [rosneuro](https://github.com/rosneuro)

## Dependencies
- torch
- neurorobotics-dl

## Example usage

### launch.xml
```xml
<?xml version="1.0"?>
<launch>
	<!--_________________________________ARGUMENTS_________________________________-->

	<!-- acquisition arguments -->
	<arg name="plugin" default='rosneuro::EGDDevice'/>	
	<arg name="devarg" default="gtec" doc="gtec 16ch"/>
	<arg name="framerate"  default='16'/>
	<arg name="samplerate" default='512'/>
			
	<!-- filter chain arguments -->
	<arg name="filter" default="ChainCfg" /> 

	<!-- classifier arguments -->
	<arg name="model_path" default="/home/curtaz/Neurorobotics/models/model.pt" />
	<arg name="model_classes" default='[769,770]' /> 

	<!--_________________________________PARAMETERS________________________________-->

	<!-- Protocol parameters -->
	<rosparam param="/protocol/subject"	 subst_value="True">$(arg subject)</rosparam>
	<rosparam param="/protocol/modality" subst_value="True">$(arg modality)</rosparam>
	<rosparam param="/protocol/task"	 subst_value="True">$(arg task)</rosparam>
	<rosparam param="/protocol/extra"	 subst_value="True">$(arg extra)</rosparam>
	
	<!-- filter chain parameters -->
	<rosparam command="load" file="$(find example)/config/ChainCfg.yaml" />

	<!--___________________________________NODES___________________________________-->
	
	<!-- aquisition node -->
	<node name="acquisition" pkg="rosneuro_acquisition" type="acquisition" output="screen" >
		<param name="~plugin" 	  value="$(arg plugin)"/>
		<param name="~devarg" 	  value="$(arg devarg)"/>
		<param name="~samplerate" value="$(arg samplerate)"/>
		<param name="~framerate"  value="$(arg framerate)"/>
	</node>
	
    <!-- filterchain node -->
	<node name="filterchain_node" pkg="rosneuro_dl" type="FilterChainNode.py" output="screen" >
			<param name="configname" value="$(arg filter)" />
	</node>

	<!-- classifier node -->
	<node name="predict_neural" pkg="rosneuro_dl" type="ClassifierNode.py" output="screen" >
		<param name="model_path" value="$(arg model_path)" />
		<param name="classes" value="$(arg model_classes)" /> 
	</node>

</launch>
```

### ChainCfg.yaml
```yaml
ChainCfg:
  samplerate: 512
  order: 2
  cutoff: [2,40]
  btype: 'band'
  lap_path: "/home/curtaz/Neurorobotics/laplacians/lapmask_antneuro_16.mat" 

```
