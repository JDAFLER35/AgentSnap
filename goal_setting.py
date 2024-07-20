import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from networkx import DiGraph, topological_sort

class Goal:
    def __init__(self, description: str, priority: float, deadline: datetime, parent_goal: 'Goal' = None):
        self.description = description
        self.priority = priority
        self.deadline = deadline
        self.parent_goal = parent_goal
        self.subgoals: List[Goal] = []
        self.progress: float = 0.0
        self.created_at = datetime.now()

    def add_subgoal(self, subgoal: 'Goal') -> None:
        self.subgoals.append(subgoal)
        subgoal.parent_goal = self

    def update_progress(self) -> None:
        if not self.subgoals:
            return
        self.progress = sum(subgoal.progress for subgoal in self.subgoals) / len(self.subgoals)

class GoalSetting:
    def __init__(self):
        self.goals: List[Goal] = []
        self.goal_graph = DiGraph()
        self.scaler = MinMaxScaler()

    def set_goal(self, description: str, priority: float, deadline: datetime, parent_goal_id: str = None) -> str:
        parent_goal = next((g for g in self.goals if id(g) == parent_goal_id), None) if parent_goal_id else None
        new_goal = Goal(description, priority, deadline, parent_goal)
        self.goals.append(new_goal)
        self.goal_graph.add_node(id(new_goal), goal=new_goal)
        
        if parent_goal:
            parent_goal.add_subgoal(new_goal)
            self.goal_graph.add_edge(id(parent_goal), id(new_goal))
        
        self._update_goal_priorities()
        return id(new_goal)

    def _update_goal_priorities(self) -> None:
        priorities = np.array([goal.priority for goal in self.goals]).reshape(-1, 1)
        normalized_priorities = self.scaler.fit_transform(priorities).flatten()
        for goal, norm_priority in zip(self.goals, normalized_priorities):
            goal.priority = norm_priority

    def get_ordered_goals(self) -> List[Goal]:
        return [self.goal_graph.nodes[node_id]['goal'] for node_id in topological_sort(self.goal_graph)]

    def update_goal_progress(self, goal_id: str, progress: float) -> None:
        goal = next((g for g in self.goals if id(g) == goal_id), None)
        if goal:
            goal.progress = progress
            while goal.parent_goal:
                goal.parent_goal.update_progress()
                goal = goal.parent_goal

    def get_next_actions(self) -> List[Dict[str, Any]]:
        current_time = datetime.now()
        actions = []
        for goal in self.get_ordered_goals():
            if goal.progress < 1.0 and goal.deadline > current_time:
                time_left = (goal.deadline - current_time).total_seconds()
                urgency = 1 / (time_left + 1)  # +1 to avoid division by zero
                importance = goal.priority * (1 - goal.progress)
                action_score = urgency * importance
                actions.append({
                    'goal_id': id(goal),
                    'description': goal.description,
                    'score': action_score,
                    'deadline': goal.deadline
                })
        return sorted(actions, key=lambda x: x['score'], reverse=True)

    def prune_completed_goals(self) -> None:
        self.goals = [goal for goal in self.goals if goal.progress < 1.0 or goal.deadline > datetime.now()]
        self.goal_graph = DiGraph()
        for goal in self.goals:
            self.goal_graph.add_node(id(goal), goal=goal)
            if goal.parent_goal:
                self.goal_graph.add_edge(id(goal.parent_goal), id(goal))

    def __str__(self) -> str:
        return f"GoalSetting with {len(self.goals)} goals"
